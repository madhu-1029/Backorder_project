from django.urls import path
from . import views


urlpatterns = [
    path("",views.home,name="home"),
    path("products/",views.products,name="products"),
    path("product/<int:product_id>/",views.product_detail,name="product_detail"),
    path("add-cart/<int:product_id>/",views.add_to_cart,name="add_to_cart"),
    path("cart/",views.cart,name="cart"),
    path("increase-cart/<int:product_id>/",views.increase_cart,name="increase_cart"),
    path("decrease-cart/<int:product_id>/",views.decrease_cart,name="decrease_cart"),
    path("remove-cart/<int:product_id>/",views.remove_from_cart,name="remove_cart"),
    path("checkout/",views.checkout,name="checkout"),
    path("orders/",views.orders,name="orders"),
    path("login/",views.login_view,name="login"),
    path("register/",views.register_view,name="register"),
    path("logout/",views.logout_view,name="logout"),
    path("admin-dashboard/",views.admin_dashboard,name="admin_dashboard"),
    path("admin-dashboard/add-product/",views.add_product,name="add_product"),
    path("admin-dashboard/add-stock/<int:product_id>/",views.add_stock,name="add_stock"),
    path("create-products/",views.create_products,name="create_products"),
    path("admin-dashboard/add-stock/",views.add_stock_page,name="add_stock_page"),
    path("admin-dashboard/delete-product/<int:product_id>/",views.delete_product,name="delete_product"),
    path("admin-dashboard/edit-product/<int:product_id>/",views.edit_product,name="edit_product"),
]