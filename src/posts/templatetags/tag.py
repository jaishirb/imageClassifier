from django import template
from PIL import Image
from posts.templatetags.main import main
#from stadistics import Estimator
register = template.Library()

var = []
@register.filter(name='driver')
def driver(url):
	global var
	#im = Image.open('..' + url)
	#salute = 'url: {}, {}'.format(url, im.size)
	salute = main('..' + url)
	var = salute 
	return "Informe: "


@register.filter(name='aux1')
def aux1(url):
	return var[0]

@register.filter(name='aux2')
def aux2(url):
	return var[1]

@register.filter(name='aux3')
def aux3(url):
	return var[2]

@register.filter(name='aux4')
def aux4(url):
	return var[3]

@register.filter(name='aux5')
def aux5(url):
	return var[4]

@register.filter(name='aux6')
def aux6(url):
	return var[5]

@register.filter(name='aux7')
def aux7(url):
	return var[6]