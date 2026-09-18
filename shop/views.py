import os
import pickle

import pandas as pd
import numpy as np

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db import transaction
from django.db.models import Sum
from django.shortcuts import render, redirect, get_object_or_404

from tensorflow.keras.models import load_model

from .models import Product, Order, OrderItem


# ============================================================
# ML FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "backorder_ann_model.keras"
)

SCALER_PATH = os.path.join(
    settings.BASE_DIR,
    "backorder_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    settings.BASE_DIR,
    "backorder_features.pkl"
)

MEDIANS_PATH = os.path.join(
    settings.BASE_DIR,
    "backorder_medians.pkl"
)


# ============================================================
# GLOBAL ML VARIABLES
# ============================================================

_model = None
_scaler = None
_feature_columns = None
_medians = None


# ============================================================
# LOAD ML FILES
# ============================================================

def load_ml_files():

    global _model
    global _scaler
    global _feature_columns
    global _medians

    try:

        if _model is None:
            _model = load_model(MODEL_PATH)

        if _scaler is None:
            with open(SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)

        if _feature_columns is None:
            with open(FEATURES_PATH, "rb") as f:
                _feature_columns = pickle.load(f)

        if _medians is None:
            with open(MEDIANS_PATH, "rb") as f:
                _medians = pickle.load(f)

        return True

    except Exception as e:

        print("ML LOAD ERROR:", e)

        return False


# ============================================================
# ANN PREDICTION
# ============================================================

def predict_product(product):

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        # If stock is ZERO, product is automatically BACKORDER
        # ----------------------------------------------------

        if product.stock <= 0:

            return "BACKORDER", 100.0

        # ----------------------------------------------------
        # Load ML files
        # ----------------------------------------------------

        if not load_ml_files():

            return "NO BACKORDER", 0.0

        # ----------------------------------------------------
        # Product data
        # ----------------------------------------------------

        data = {

            "national_inv": product.stock,

            "lead_time": product.lead_time,

            "in_transit_qty": product.in_transit_qty,

            "forecast_3_month": product.forecast_3_month,

            "forecast_6_month": product.forecast_6_month,

            "forecast_9_month": product.forecast_9_month,

            "sales_1_month": product.sales_1_month,

            "sales_3_month": product.sales_3_month,

            "min_bank": product.min_bank,

            "pieces_past_due": product.pieces_past_due,

            "perf_6_month_avg": product.perf_6_month_avg,

            "perf_12_month_avg": product.perf_12_month_avg,

            "local_bo_qty": 0,

            "potential_issue": 0,

            "deck_risk": 0,

            "oe_constraint": 0,

            "ppap_risk": 0,

            "stop_auto_buy": 0,

            "rev_stop": 0,

        }

        # ----------------------------------------------------
        # Feature columns
        # ----------------------------------------------------

        if isinstance(_feature_columns, np.ndarray):

            feature_columns = list(_feature_columns)

        else:

            feature_columns = list(_feature_columns)

        # ----------------------------------------------------
        # Create input row
        # ----------------------------------------------------

        row = {}

        for column in feature_columns:

            if column in data:

                row[column] = data[column]

            else:

                if isinstance(_medians, dict):

                    row[column] = _medians.get(
                        column,
                        0
                    )

                elif hasattr(_medians, "get"):

                    try:

                        row[column] = _medians.get(
                            column,
                            0
                        )

                    except Exception:

                        row[column] = 0

                else:

                    row[column] = 0

        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame(
            [row],
            columns=feature_columns
        )

        # ----------------------------------------------------
        # Convert values to numeric
        # ----------------------------------------------------

        for column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.fillna(0)

        # ----------------------------------------------------
        # Scale input
        # ----------------------------------------------------

        X_scaled = _scaler.transform(df)

        # ----------------------------------------------------
        # ANN prediction
        # ----------------------------------------------------

        probability = float(
            _model.predict(
                X_scaled,
                verbose=0
            )[0][0]
        )

        probability_percent = round(
            probability * 100,
            2
        )

        # ----------------------------------------------------
        # Prediction result
        # ----------------------------------------------------

        if probability >= 0.5:

            prediction = "BACKORDER"

        else:

            prediction = "NO BACKORDER"

        return prediction, probability_percent

    except Exception as e:

        print("ANN PREDICTION ERROR:", e)

        return "NO BACKORDER", 0.0


# ============================================================
# HOME
# ============================================================

def home(request):

    products = Product.objects.all()[:10]

    return render(
        request,
        "home.html",
        {
            "products": products
        }
    )


# ============================================================
# PRODUCTS
# ============================================================

def products(request):

    products = Product.objects.all()

    return render(
        request,
        "products.html",
        {
            "products": products
        }
    )


# ============================================================
# PRODUCT DETAIL
# ============================================================

def product_detail(request, product_id):

    product = get_object_or_404(
        Product,
        id=product_id
    )

    return render(
        request,
        "product_detail.html",
        {
            "product": product
        }
    )


# ============================================================
# ADD TO CART
# ============================================================

def add_to_cart(request, product_id):

    product = get_object_or_404(
        Product,
        id=product_id
    )

    if product.stock <= 0:

        messages.error(
            request,
            f"{product.name} is out of stock."
        )

        return redirect("products")

    cart = request.session.get(
        "cart",
        {}
    )

    product_id = str(product_id)

    if product_id in cart:

        if cart[product_id] < product.stock:

            cart[product_id] += 1

        else:

            messages.warning(
                request,
                f"Only {product.stock} "
                f"{product.name} available."
            )

    else:

        cart[product_id] = 1

    request.session["cart"] = cart

    request.session.modified = True

    return redirect("cart")


# ============================================================
# CART
# ============================================================

def cart(request):

    cart_data = request.session.get(
        "cart",
        {}
    )

    cart_products = []

    total = 0

    for product_id, quantity in cart_data.items():

        try:

            product = Product.objects.get(
                id=product_id
            )

        except Product.DoesNotExist:

            continue

        if quantity <= 0:
            continue

        subtotal = (
            product.price * quantity
        )

        cart_products.append({

            "product": product,

            "quantity": quantity,

            "subtotal": subtotal,

        })

        total += subtotal

    return render(
        request,
        "cart.html",
        {
            "cart_products": cart_products,
            "total": total,
        }
    )


# ============================================================
# INCREASE CART
# ============================================================

def increase_cart(request, product_id):

    product = get_object_or_404(
        Product,
        id=product_id
    )

    cart = request.session.get(
        "cart",
        {}
    )

    product_id = str(product_id)

    if product_id in cart:

        if cart[product_id] < product.stock:

            cart[product_id] += 1

        else:

            messages.warning(
                request,
                f"Only {product.stock} "
                f"{product.name} available."
            )

    request.session["cart"] = cart

    request.session.modified = True

    return redirect("cart")


# ============================================================
# DECREASE CART
# ============================================================

def decrease_cart(request, product_id):

    cart = request.session.get(
        "cart",
        {}
    )

    product_id = str(product_id)

    if product_id in cart:

        if cart[product_id] > 1:

            cart[product_id] -= 1

        else:

            del cart[product_id]

    request.session["cart"] = cart

    request.session.modified = True

    return redirect("cart")


# ============================================================
# REMOVE FROM CART
# ============================================================

def remove_from_cart(request, product_id):

    cart = request.session.get(
        "cart",
        {}
    )

    product_id = str(product_id)

    if product_id in cart:

        del cart[product_id]

    request.session["cart"] = cart

    request.session.modified = True

    return redirect("cart")


# ============================================================
# CHECKOUT
# ============================================================

@login_required
def checkout(request):

    cart_data = request.session.get(
        "cart",
        {}
    )

    if not cart_data:

        messages.warning(
            request,
            "Your cart is empty."
        )

        return redirect("products")

    products_to_buy = []

    for product_id, quantity in cart_data.items():

        product = get_object_or_404(
            Product,
            id=product_id
        )

        if quantity <= 0:
            continue

        if product.stock < quantity:

            messages.error(
                request,
                f"Not enough stock for "
                f"{product.name}. "
                f"Available: {product.stock}"
            )

            return redirect("cart")

        products_to_buy.append(
            (product, quantity)
        )

    if not products_to_buy:

        messages.warning(
            request,
            "Your cart is empty."
        )

        return redirect("cart")

    # --------------------------------------------------------
    # Calculate total
    # --------------------------------------------------------

    total_amount = 0

    for product, quantity in products_to_buy:

        total_amount += (
            product.price * quantity
        )

    # --------------------------------------------------------
    # Create order
    # --------------------------------------------------------

    with transaction.atomic():

        order = Order.objects.create(

            user=request.user,

            total_amount=total_amount

        )

        backorder_products = []

        for product, quantity in products_to_buy:

            # Create order item

            OrderItem.objects.create(

                order=order,

                product=product,

                quantity=quantity,

                price=product.price

            )

            # Old stock

            old_stock = product.stock

            # Reduce stock

            product.stock = (
                product.stock - quantity
            )

            # Increase sales

            product.sales_1_month += quantity

            product.sales_3_month += quantity

            # ------------------------------------------------
            # ANN Prediction
            # ------------------------------------------------

            prediction, probability = (
                predict_product(product)
            )

            product.prediction = prediction

            product.prediction_probability = probability

            product.save()

            # ------------------------------------------------
            # Backorder list
            # ------------------------------------------------

            if prediction == "BACKORDER":

                backorder_products.append(
                    product.name
                )

            print(
                f"{product.name}: "
                f"{old_stock} - {quantity} = "
                f"{product.stock}"
            )

    # --------------------------------------------------------
    # Clear cart
    # --------------------------------------------------------

    request.session["cart"] = {}

    request.session.modified = True

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    if backorder_products:

        names = ", ".join(
            backorder_products
        )

        messages.warning(
            request,
            f"Order placed successfully. "
            f"Backorder detected for: {names}"
        )

    else:

        messages.success(
            request,
            "Order placed successfully!"
        )

    return redirect("orders")


# ============================================================
# ORDERS
# ============================================================

@login_required
def orders(request):

    orders = Order.objects.filter(
        user=request.user
    ).prefetch_related(
        "items"
    ).order_by(
        "-created_at"
    )

    return render(
        request,
        "orders.html",
        {
            "orders": orders
        }
    )


# ============================================================
# LOGIN
# ============================================================

def login_view(request):

    if request.user.is_authenticated:

        if request.user.is_staff:

            return redirect(
                "admin_dashboard"
            )

        return redirect("home")

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        password = request.POST.get(
            "password"
        )

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            if user.is_staff:

                return redirect(
                    "admin_dashboard"
                )

            return redirect("home")

        messages.error(
            request,
            "Invalid username or password."
        )

    return render(
        request,
        "login.html"
    )


# ============================================================
# REGISTER
# ============================================================

def register_view(request):

    if request.user.is_authenticated:

        return redirect("home")

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        email = request.POST.get(
            "email"
        )

        password = request.POST.get(
            "password"
        )

        confirm_password = request.POST.get(
            "confirm_password"
        )

        if not username or not password:

            messages.error(
                request,
                "Please fill all required fields."
            )

            return redirect("register")

        if password != confirm_password:

            messages.error(
                request,
                "Passwords do not match."
            )

            return redirect("register")

        if User.objects.filter(
            username=username
        ).exists():

            messages.error(
                request,
                "Username already exists."
            )

            return redirect("register")

        user = User.objects.create_user(

            username=username,

            email=email,

            password=password

        )

        login(
            request,
            user
        )

        return redirect("home")

    return render(
        request,
        "register.html"
    )


# ============================================================
# LOGOUT
# ============================================================

def logout_view(request):

    logout(request)

    return redirect("login")


# ============================================================
# CREATE SAMPLE PRODUCTS
# ============================================================

def create_products(request):

    products_data = [

        (
            "Toy",
            "Kids",
            500,
            100
        ),

        (
            "Helmet",
            "Safety",
            1200,
            100
        ),

        (
            "Backpack",
            "Fashion",
            900,
            100
        ),

        (
            "Water Bottle",
            "Home",
            300,
            100
        ),

        (
            "Smartphone",
            "Electronics",
            15000,
            100
        ),

        (
            "Smart Watch",
            "Electronics",
            2500,
            100
        ),

        (
            "Headphones",
            "Electronics",
            1500,
            100
        ),

        (
            "Sports Shoes",
            "Fashion",
            1800,
            100
        ),

        (
            "Hand Bag",
            "Fashion",
            1200,
            100
        ),

        (
            "Laptop",
            "Electronics",
            55000,
            50
        ),

        (
            "Cricket Bat",
            "Sports",
            2500,
            100
        ),

    ]

    for name, category, price, stock in products_data:

        Product.objects.get_or_create(

            name=name,

            defaults={

                "category": category,

                "price": price,

                "stock": stock,

                "description":
                    f"Good quality {name}",

                "lead_time": 10,

                "in_transit_qty": 20,

                "forecast_3_month": 50,

                "forecast_6_month": 100,

                "forecast_9_month": 150,

                "sales_1_month": 10,

                "sales_3_month": 30,

                "min_bank": 20,

                "pieces_past_due": 0,

                "perf_6_month_avg": 0.8,

                "perf_12_month_avg": 0.8,

                "prediction":
                    "NO BACKORDER",

                "prediction_probability":
                    0,

            }

        )

    messages.success(
        request,
        "Sample products created successfully!"
    )

    return redirect("products")


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@login_required
def admin_dashboard(request):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    products = Product.objects.all().order_by("id")

    # --------------------------------------------------------
    # Dashboard statistics
    # --------------------------------------------------------

    total_products = products.count()

    total_orders = Order.objects.count()

    backorder_products = products.filter(
        prediction="BACKORDER"
    ).count()

    low_stock_products = products.filter(
        stock__lt=20
    ).count()

    products_data = []

    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    for product in products:

        # Calculate total sold

        sold = OrderItem.objects.filter(
            product=product
        ).aggregate(
            total=Sum("quantity")
        )["total"] or 0

        # ----------------------------------------------------
        # Stock status
        # ----------------------------------------------------

        if product.stock <= 0:

            stock_status = "OUT OF STOCK"

        elif product.stock < 20:

            stock_status = "LOW STOCK"

        else:

            stock_status = "IN STOCK"

        # ----------------------------------------------------
        # Prediction status
        # ----------------------------------------------------

        # IMPORTANT:
        # Stock zero means BACKORDER

        if product.stock <= 0:

            prediction_status = "BACKORDER"

            # Keep database prediction correct too

            if product.prediction != "BACKORDER":

                product.prediction = "BACKORDER"

                product.prediction_probability = 100

                product.save(
                    update_fields=[
                        "prediction",
                        "prediction_probability"
                    ]
                )

        elif product.prediction == "BACKORDER":

            prediction_status = "BACKORDER"

        else:

            prediction_status = "NO BACKORDER"

        # ----------------------------------------------------
        # Add data
        # ----------------------------------------------------

        products_data.append({

            "product": product,

            "sold": sold,

            "stock_status": stock_status,

            "prediction_status":
                prediction_status,

        })

    return render(

        request,

        "admin_dashboard.html",

        {

            "products_data":
                products_data,

            "total_products":
                total_products,

            "total_orders":
                total_orders,

            "backorder_products":
                backorder_products,

            "low_stock_products":
                low_stock_products,

        }

    )


# ============================================================
# ADD PRODUCT
# ============================================================

@login_required
def add_product(request):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    if request.method == "POST":

        name = request.POST.get(
            "name"
        )

        category = request.POST.get(
            "category"
        )

        description = request.POST.get(
            "description"
        )

        price = request.POST.get(
            "price"
        )

        stock = request.POST.get(
            "stock"
        )

        lead_time = (
            request.POST.get("lead_time")
            or 10
        )

        in_transit_qty = (
            request.POST.get("in_transit_qty")
            or 20
        )

        forecast_3_month = (
            request.POST.get(
                "forecast_3_month"
            )
            or 50
        )

        forecast_6_month = (
            request.POST.get(
                "forecast_6_month"
            )
            or 100
        )

        forecast_9_month = (
            request.POST.get(
                "forecast_9_month"
            )
            or 150
        )

        min_bank = (
            request.POST.get(
                "min_bank"
            )
            or 20
        )

        if not name or not price or not stock:

            messages.error(
                request,
                "Product name, price and stock are required."
            )

            return redirect(
                "add_product"
            )

        product = Product.objects.create(

            name=name,

            category=category or "General",

            description=description or "",

            price=price,

            stock=stock,

            lead_time=lead_time,

            in_transit_qty=in_transit_qty,

            forecast_3_month=
                forecast_3_month,

            forecast_6_month=
                forecast_6_month,

            forecast_9_month=
                forecast_9_month,

            sales_1_month=0,

            sales_3_month=0,

            min_bank=min_bank,

            pieces_past_due=0,

            perf_6_month_avg=0.8,

            perf_12_month_avg=0.8,

            prediction=
                "NO BACKORDER",

            prediction_probability=0

        )

        # ----------------------------------------------------
        # If new product stock is zero
        # ----------------------------------------------------

        if product.stock <= 0:

            product.prediction = "BACKORDER"

            product.prediction_probability = 100

            product.save()

        messages.success(

            request,

            f"{product.name} added successfully!"

        )

        return redirect(
            "admin_dashboard"
        )

    return render(
        request,
        "add_product.html"
    )


# ============================================================
# ADD STOCK
# ============================================================

@login_required
def add_stock(request, product_id):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    product = get_object_or_404(
        Product,
        id=product_id
    )

    if request.method == "POST":

        quantity = request.POST.get(
            "quantity"
        )

        try:

            quantity = int(quantity)

            if quantity <= 0:

                messages.error(
                    request,
                    "Please enter a valid quantity."
                )

                return redirect(
                    "add_stock",
                    product_id=product.id
                )

            old_stock = product.stock

            # Add stock

            product.stock += quantity

            # Recalculate ANN prediction

            prediction, probability = (
                predict_product(product)
            )

            product.prediction = prediction

            product.prediction_probability = (
                probability
            )

            product.save()

            print(

                f"{product.name}: "
                f"{old_stock} + {quantity} = "
                f"{product.stock}"

            )

            messages.success(

                request,

                f"{quantity} stock added to "
                f"{product.name}. "
                f"Current stock: {product.stock}"

            )

            return redirect(
                "admin_dashboard"
            )

        except (
            ValueError,
            TypeError
        ):

            messages.error(
                request,
                "Please enter a valid number."
            )

            return redirect(
                "add_stock",
                product_id=product.id
            )

    return render(

        request,

        "add_stock.html",

        {
            "product": product
        }

    )


# ============================================================
# ADD STOCK PAGE
# ============================================================

@login_required
def add_stock_page(request):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    products = Product.objects.all().order_by(
        "name"
    )

    return render(

        request,

        "add_stock_page.html",

        {
            "products": products
        }

    )


# ============================================================
# DELETE PRODUCT
# ============================================================

@login_required
def delete_product(request, product_id):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    product = get_object_or_404(
        Product,
        id=product_id
    )

    if request.method == "POST":

        product_name = product.name

        product.delete()

        messages.success(

            request,

            f"{product_name} deleted successfully!"

        )

        return redirect(
            "admin_dashboard"
        )

    return render(

        request,

        "delete_product.html",

        {
            "product": product
        }

    )


# ============================================================
# EDIT PRODUCT
# ============================================================

@login_required
def edit_product(request, product_id):

    if not request.user.is_staff:

        messages.error(
            request,
            "Admin access only."
        )

        return redirect("home")

    product = get_object_or_404(
        Product,
        id=product_id
    )

    if request.method == "POST":

        product.name = request.POST.get(
            "name"
        )

        product.category = request.POST.get(
            "category"
        )

        product.description = request.POST.get(
            "description"
        )

        product.price = request.POST.get(
            "price"
        )

        product.stock = request.POST.get(
            "stock"
        )

        product.lead_time = (
            request.POST.get(
                "lead_time"
            )
            or 10
        )

        product.in_transit_qty = (
            request.POST.get(
                "in_transit_qty"
            )
            or 0
        )

        product.forecast_3_month = (
            request.POST.get(
                "forecast_3_month"
            )
            or 0
        )

        product.forecast_6_month = (
            request.POST.get(
                "forecast_6_month"
            )
            or 0
        )

        product.forecast_9_month = (
            request.POST.get(
                "forecast_9_month"
            )
            or 0
        )

        product.sales_1_month = (
            request.POST.get(
                "sales_1_month"
            )
            or 0
        )

        product.sales_3_month = (
            request.POST.get(
                "sales_3_month"
            )
            or 0
        )

        product.min_bank = (
            request.POST.get(
                "min_bank"
            )
            or 0
        )

        product.pieces_past_due = (
            request.POST.get(
                "pieces_past_due"
            )
            or 0
        )

        product.perf_6_month_avg = (
            request.POST.get(
                "perf_6_month_avg"
            )
            or 0
        )

        product.perf_12_month_avg = (
            request.POST.get(
                "perf_12_month_avg"
            )
            or 0
        )

        # ----------------------------------------------------
        # Recalculate prediction
        # ----------------------------------------------------

        prediction, probability = (
            predict_product(product)
        )

        product.prediction = prediction

        product.prediction_probability = (
            probability
        )

        product.save()

        messages.success(

            request,

            f"{product.name} updated successfully!"

        )

        return redirect(
            "admin_dashboard"
        )

    return render(

        request,

        "edit_product.html",

        {
            "product": product
        }

    )