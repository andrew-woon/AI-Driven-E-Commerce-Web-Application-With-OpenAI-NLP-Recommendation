from django.contrib import admin
from django.urls import path , include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.main.urls')),
    path('accounts/', include('apps.accounts.urls')),    
    path('seller_accounts/', include('apps.seller_accounts.urls')),
    path('cart/', include('apps.cart.urls'))
]


urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

print(urlpatterns)

handler404 = 'apps.main.views.error_404_view'