from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.decorators import api_view
import json
from datetime import date
from . import utils

# Create your views here.
def home(request):
    ticker_list = ['AAPL','MSFT','AMZN','GOOG','TSLA','FB','NVDA','PYPL','NFLX','CMCSA']
    return render(request,'home.html',{'ticker_list': ticker_list})

def graph(request):
    stock = request.GET['stock_name']
    
    graph1 = utils.close_price(stock)
    graph2 = utils.candle_chart(stock)
    predicted_price = utils.predicted_price(stock)

    return render(request,'graph.html',{'stock':stock,'graph':graph1,'graph2':graph2,'predicted_price':predicted_price})

@api_view(['GET'])
def rest_api(request):
    stock = request.GET['stock']
    today = date.today()
    price = {'date':today,stock:utils.predicted_price(stock)}
    return HttpResponse( json.dumps(str(price) ))
