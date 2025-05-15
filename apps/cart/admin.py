from django.contrib import admin
from .models import Sales, Cart, CartProduct, Order, Product, ProductImagesFiles


admin.site.register([Sales, Cart, CartProduct, Order, Product, ProductImagesFiles])
# Register your models here.
