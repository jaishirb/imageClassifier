from django import template

register = template.Library()

@register.filter(name='hello_world')
def hello_world(name):
    salute = 'Hello' 
    return salute