from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('ner_extraction_app/', include('ner_extraction_app.urls')),
    path('admin/', admin.site.urls),
]