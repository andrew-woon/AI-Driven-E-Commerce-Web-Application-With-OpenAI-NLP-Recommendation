# Django imports
from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.urls import reverse
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse
from django.db.models import Q, Min, Max
from django.template.loader import render_to_string
from django.core.paginator import Paginator
# Core Python & third-party
import os
import sys
import json
import traceback
import inspect
import numpy as np
import pandas as pd
import pickle
# TensorFlow
from tensorflow.keras.models import load_model
# App-specific imports
from apps.main.models import Product
from apps.cart.models import *
from apps.seller_accounts.models import *
from openai import OpenAI
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import os
from django.conf import settings
client = OpenAI(api_key="your-openai-api-key")

# Load once globally
MODEL_PATH = os.path.join(settings.BASE_DIR, 'apps','main', 'model_data', 'multioutput_fashion_model.h5')
ENCODER_PATH = os.path.join(settings.BASE_DIR, 'apps','main', 'model_data', 'label_encoders.pkl')
CSV_PATH = os.path.join(settings.BASE_DIR, 'apps','main', 'model_data', 'enhanced_fashion_dataset_300.csv')

model = load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

le_gender = encoders['gender']
le_subcat = encoders['subcategory']
le_article = encoders['article_type']

original_columns = pd.get_dummies(
    pd.read_csv(CSV_PATH)[[
        'age_group', 'style_preference', 'color_preference', 'shopping_frequency',
        'spending_category', 'shopping_occasion', 'interest', 'fit_preference',
        'fabric_preference', 'season_shopping'
    ]]
).columns.tolist()
def ml_recommend_view(request):
    # set_page_active('Recommendation')
    product_list = Product.objects.none()
    ai_recommendation = None
    ai_raw_response = None
    ai_gender = ai_sub_category = ai_article_type = None

    if request.method == "POST":
        try:
            # Get form inputs
            user_input = {
                'age_group': request.POST.get('age_group'),
                'style_preference': request.POST.get('style_preference'),
                'color_preference': request.POST.get('color_preference'),
                'shopping_frequency': request.POST.get('shopping_frequency'),
                'spending_category': request.POST.get('spending_category'),
                'shopping_occasion': request.POST.get('shopping_occasion'),
                'interest': request.POST.get('interest'),
                'fit_preference': request.POST.get('fit_preference'),
                'fabric_preference': request.POST.get('fabric_preference'),
                'season_shopping': request.POST.get('season_shopping'),
                # 'recent_item': request.POST.get('recent_item'),
            }

            # Encode input
            user_df = pd.DataFrame([user_input])
            user_encoded = pd.get_dummies(user_df)

            for col in original_columns:
                if col not in user_encoded:
                    user_encoded[col] = 0

            user_encoded = user_encoded[original_columns].astype('float32')

            # Predict using local model
            pred_gender, pred_subcat, pred_article = model.predict(user_encoded)
            ai_gender = le_gender.inverse_transform([np.argmax(pred_gender)])[0]
            ai_sub_category = le_subcat.inverse_transform([np.argmax(pred_subcat)])[0]
            ai_article_type = le_article.inverse_transform([np.argmax(pred_article)])[0]

            ai_recommendation = {
                "gender": ai_gender,
                "sub_category": ai_sub_category,
                "article_type": ai_article_type
            }

            ai_raw_response = json.dumps(ai_recommendation, indent=2)

            # Filter recommended products
            if ai_gender and ai_sub_category and ai_article_type:
                product_list = Product.objects.filter(
                    (Q(gender_cat__iexact=ai_gender) & Q(sub_cat__iexact=ai_sub_category)) |
                    (Q(gender_cat__iexact=ai_gender) & Q(articel_type__iexact=ai_article_type)) |
                    (Q(sub_cat__iexact=ai_sub_category) & Q(articel_type__iexact=ai_article_type))
                ).distinct()

                for product in product_list:
                    match_count = 0
                    if product.gender_cat and product.gender_cat.lower() == ai_gender.lower():
                        match_count += 1
                    if product.sub_cat and product.sub_cat.lower() == ai_sub_category.lower():
                        match_count += 1
                    if product.articel_type and product.articel_type.lower() == ai_article_type.lower():
                        match_count += 1
                    product.match_count = match_count

        except Exception as e:
            traceback.print_exc()
            messages.error(request, "Error processing your request. Please try again.")

    return render(request, 'ml_recom.html', {
        'product_list': product_list,
        'ai_recommendation': ai_recommendation,
        'ai_raw_response': ai_raw_response,
        'ai_gender': ai_gender,
        'ai_sub_category': ai_sub_category,
        'ai_article_type': ai_article_type,
    })
def chat_recommend_view(request):
    # set_page_active('Recommendation')
    product_list = Product.objects.none()  # default to empty
    ai_recommendation = None
    ai_raw_response = None

    # These are passed to template for dynamic filtering/display
    ai_gender = None
    ai_sub_category = None
    ai_article_type = None

    if request.method == "POST":
        user_input = request.POST.get('user_input', '').strip()

        if user_input:
            prompt = f"""
You are an assistant that only returns JSON.

Analyze the following shopping interest and return exactly this format, and nothing else:
{{
  "gender": "Men/Women/Kids",
  "sub_category": "Category like Footwear, Jackets, etc.",
  "article_type": "Specific type like Sneakers, Leather Shoes, Hoodie, etc."
}}

User input: "{user_input}"
"""
            try:
                # response = client.chat.completions.create(
                #     model="gpt-4o",
                #     messages=[{"role": "user", "content": prompt}],
                #     temperature=0
                # )
                # ai_message = response.choices[0].message.content

                # Use fake data for testing
                ai_message = '{"gender": "Men", "sub_category": "FootWear", "article_type": "Tshirts"}'

                ai_raw_response = ai_message
                ai_recommendation = json.loads(ai_message)

                # Extract individual values
                ai_gender = ai_recommendation.get('gender')
                ai_sub_category = ai_recommendation.get('sub_category')
                ai_article_type = ai_recommendation.get('article_type')

                if ai_gender and ai_sub_category and ai_article_type:
                    product_list = Product.objects.filter(
                        (Q(gender_cat__iexact=ai_gender) & Q(sub_cat__iexact=ai_sub_category)) |
                        (Q(gender_cat__iexact=ai_gender) & Q(articel_type__iexact=ai_article_type)) |
                        (Q(sub_cat__iexact=ai_sub_category) & Q(articel_type__iexact=ai_article_type))
                    ).distinct()

                    # Now add match counts manually
                    for product in product_list:
                        match_count = 0

                        if ai_gender and product.gender_cat and product.gender_cat.lower() == ai_gender.lower():
                            match_count += 1

                        if ai_sub_category and product.sub_cat and product.sub_cat.lower() == ai_sub_category.lower():
                            match_count += 1

                        if ai_article_type and product.articel_type and product.articel_type.lower() == ai_article_type.lower():
                            match_count += 1

                        product.match_count = match_count


            except Exception as e:
                import traceback
                traceback.print_exc()
                messages.error(request, "Error processing your request. Please try again.")

    return render(request, 'chatgp_recom.html', {
        'product_list': product_list,
        'ai_recommendation': ai_recommendation,
        'ai_raw_response': ai_raw_response,
        'ai_gender': ai_gender,
        'ai_sub_category': ai_sub_category,
        'ai_article_type': ai_article_type,
    })
def index(request):
    # set_page_active('Home')
    product_list = Product.objects.all()
    # paginator = Paginator(product_list, 2)
    # page_number = request.GET.get('page')
    # product_list = paginator.get_page(page_number)
    # allcat = Category.objects.all()
    return render(request, "index.html", {'product_list': product_list})

def show_all_products(request):
    # set_page_active('Shop')
    product_list = Product.objects.all().order_by('id')
    minprice = Product.objects.all().aggregate(Min('market_price'))['market_price__min']
    maxprice = Product.objects.all().aggregate(Max('market_price'))['market_price__max']

    gender_cat = Product.objects.values_list('gender_cat',flat=True).distinct()
    sub_cat = Product.objects.values_list('sub_cat',flat=True).distinct()
    articel_type = Product.objects.values_list('articel_type',flat=True).distinct()

    color = Product.objects.values_list('color',flat=True).distinct()
    brand = Product.objects.values_list('brand',flat=True).distinct()

    paginator = Paginator(product_list, 15)
    page_number = request.GET.get('page')
    product_list = paginator.get_page(page_number)
    return render(request, "shop-grid-ls.html", {'product_list': product_list,'gender_cat':list(gender_cat),'sub_category':list(sub_cat),'article_type':list(articel_type), 'minprice':minprice, 'maxprice':maxprice, 'colors':list(color),'brands':list(brand)})

def category(request):
    # allcat = Category.objects.all()
    productcount = request.session.get('productcount')
    return render(request, "index.html", {'productcount':productcount})

def Search_Result(request, keyword):
    product_list = Product.objects.filter(Q(title__icontains = keyword) | Q(description__icontains = keyword))

    minprice = product_list.aggregate(Min('market_price'))['market_price__min']
    maxprice = product_list.aggregate(Max('market_price'))['market_price__max']

    gender_cat = product_list.values_list('gender_cat',flat=True).distinct()
    sub_cat = product_list.values_list('sub_cat',flat=True).distinct()
    articel_type = product_list.values_list('articel_type',flat=True).distinct()

    color = product_list.values_list('color',flat=True).distinct()
    brand = product_list.values_list('brand',flat=True).distinct()
    
    paginator = Paginator(product_list, 15)
    page_number = request.GET.get('page')
    product_list = paginator.get_page(page_number)

    return render(request, 'search_result.html', {'keyword':keyword, 'search_result': product_list,'gender_cat':list(gender_cat),'sub_category':list(sub_cat),'article_type':list(articel_type), 'minprice':minprice, 'maxprice':maxprice, 'colors':list(color),'brands':list(brand)})

def Search_Product(request):
    product_list = Product.objects.all()
    if request.method == "POST":
        keyword = request.POST.get('keyword')
        # return redirect('search_result'+ keyword)
        return HttpResponseRedirect(reverse('apps.main:search_result', args=[keyword]))

def Single_Product(request, pid):
   single_product = Product.objects.get(id = pid)
   return render(request, 'product_details.html', {'single_product':single_product}) 

def filter_data(request):
    product_all_list = Product.objects.all().order_by('id')
    product_list = filter_data_functionality(request,product_all_list)
    template = render_to_string('ajax/product-list.html', {'product_list': product_list})
    return JsonResponse({'data':template})

def filter_search_data(request):
    product_list = filter_data_functionality(request)
    template = render_to_string('ajax/product-list_search_result.html', {'product_list': product_list})
    return JsonResponse({'data_search':template})
def filter_auto(request):
    productlist=[]
    products = request.GET.get("prod")
    if not products:
        products=''
    
    productlist = list(products.split("-"))
    gender_categories= productlist[1:2]
    sub_categories= productlist[2:3]
    article_categories= productlist[3:4]
   
    

    product_list = Product.objects.all().order_by('id')
    
    if len(gender_categories)>0:
        # category = Category.objects.filter(title__in=categories)
        product_list = product_list.filter(gender_cat__in=gender_categories)
    if len(sub_categories)>0:
        product_list = product_list.filter(sub_cat__in=sub_categories)
    if len(article_categories)>0:
        product_list = product_list.filter(articel_type__in=article_categories)
    
    # set_page_active('Shop')
    minprice = Product.objects.all().aggregate(Min('market_price'))['market_price__min']
    maxprice = Product.objects.all().aggregate(Max('market_price'))['market_price__max']
    subCategory_list = ['Topwear', 'Bottomwear', 'Dress', 'Saree', 'Shoes', 'Innerwear', 'Headwear', 'Socks', 'Flip Flops']
    all_articel_type_list = ['Belts', 'Blazers', 'Booties', 'Boxers', 'Bra', 'Briefs', 'Camisoles', 'Capris', 'Caps', 'Casual Shoes', 'Churidar', 'Dresses', 'Dupatta', 'Flats', 'Flip Flops', 'Formal Shoes', 'Hat', 'Headband', 'Heels', 'Innerwear Vests', 'Jackets', 'Jeans', 'Jeggings', 'Jumpsuit', 'Kurtas', 'Kurtis', 'Leggings', 'Lehenga Choli', 'Nehru Jackets', 'Patiala', 'Rain Jacket', 'Rain Trousers', 'Rompers', 'Salwar', 'Salwar and Dupatta', 'Sandals', 'Sarees', 'Shapewear', 'Shirts', 'Shorts', 'Shrug', 'Skirts', 'Socks', 'Sports Shoes', 'Stockings', 'Suits', 'Suspenders', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tights', 'Tops', 'Track Pants', 'Tracksuits', 'Trousers', 'Trunk', 'Tshirts', 'Tunics', 'Waistcoat']
    # all_category = Category.objects.all()
    gender_cat = Product.objects.values_list('gender_cat',flat=True).distinct()
    sub_cat = Product.objects.values_list('sub_cat',flat=True).distinct()
    articel_type = Product.objects.values_list('articel_type',flat=True).distinct()
    color = Product.objects.values_list('color',flat=True).distinct()
    paginator = Paginator(product_list, 6)
    page_number = request.GET.get('page')
    product_list = paginator.get_page(page_number)
    return render(request, "shop-grid-ls.html", {'product_list': product_list,'gender_cat':list(gender_cat),'sub_category':list(sub_cat),'article_type':list(articel_type), 'minprice':minprice, 'maxprice':maxprice, 'colors':list(color)})
    
def reviews(request, prodid):
    # set_page_active('Shop')
    user=User.objects.get(id=request.user.id)
    product=Product.objects.get(id=prodid)
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        rating = request.POST.get('rating')
        review = request.POST.get('review')
        pros = request.POST.get('pros')
        cons = request.POST.get('cons')
        Reviews.objects.create(product=product, user=user, name=name, email=email, rating=rating, review=review, pros=pros, cons=cons)
        messages.success(request,'Your Reviews are Submitted')
    return redirect('apps.main:single_product', prodid)
          
def error_404_view(request, exception):
    return render(request, '404-illustration.html')