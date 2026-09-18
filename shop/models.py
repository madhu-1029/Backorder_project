from django.db import models
from django.contrib.auth.models import User


class Product(models.Model):

    name = models.CharField(max_length=200)

    category = models.CharField(
        max_length=100,
        default="General"
    )

    description = models.TextField(
        blank=True
    )

    price = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )

    stock = models.PositiveIntegerField(
        default=0
    )

    # ML Features

    lead_time = models.FloatField(
        default=10
    )

    in_transit_qty = models.FloatField(
        default=0
    )

    forecast_3_month = models.FloatField(
        default=0
    )

    forecast_6_month = models.FloatField(
        default=0
    )

    forecast_9_month = models.FloatField(
        default=0
    )

    sales_1_month = models.FloatField(
        default=0
    )

    sales_3_month = models.FloatField(
        default=0
    )

    min_bank = models.FloatField(
        default=0
    )

    pieces_past_due = models.FloatField(
        default=0
    )

    perf_6_month_avg = models.FloatField(
        default=0
    )

    perf_12_month_avg = models.FloatField(
        default=0
    )

    # ANN Result

    prediction = models.CharField(
        max_length=30,
        default="NO BACKORDER"
    )

    prediction_probability = models.FloatField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.name


class Order(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    total_amount = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"Order #{self.id}"


class OrderItem(models.Model):

    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name="items"
    )

    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE
    )

    quantity = models.PositiveIntegerField()

    price = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )

    def __str__(self):
        return self.product.name