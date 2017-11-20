from django import forms
from django.contrib.auth import (
	authenticate,
	get_user_model,
	login,
	logout,
	)

User = get_user_model()

class UserLoginForm(forms.Form):
	username = forms.CharField()
	password = forms.CharField(widget=forms.PasswordInput)

	def clean(self, *args, **kwargs):
		username = self.cleaned_data.get("username")
		password = self.cleaned_data.get("password")
		user_qs = User.objects.filter(username=username)
		if username and password:
			user = authenticate(username=username, password=password)
			if user_qs.count()==1:
				user = user_qs.first()
			if not user:
				raise forms.ValidationError("Este usuario no existe")
			if not user.check_password(password):
				raise forms.ValidationError("Contraseña incorrecta")
#			if not user.is_activate:
#				raise forms.ValidationError("Este usuario no está activo")
			return super(UserLoginForm, self).clean(*args, **kwargs)

class UserRegisterForm(forms.ModelForm):
	email = forms.EmailField(label='Dirección de correo electrónico')
	email2 = forms.EmailField(label='Confirmar correo electrónico')
	password = forms.CharField(widget=forms.PasswordInput)

	class Meta:
		model = User
		fields = [
			'username',
			'email',
			'email2',
			'password'
		]

	# def clean(self):
	# 	email = self.cleaned_data.get('email')
	# 	email2 = self.cleaned_data.get('email2')
	# 	if email != email2:
	# 		raise forms.ValidationError("Los correos no coinciden")
	# 	email_qs = User.objects.filter(email=email)
	# 	if email_qs.exist():
	# 		raise forms.ValidationError("Este correo ya está en uso")
	# 	return super(UserRegisterForm, self).clean(*args, **kwargs)

	def clean_email2(self):
		email = self.cleaned_data.get('email')
		email2 = self.cleaned_data.get('email2')
		if email != email2:
			raise forms.ValidationError("Los correos no coinciden")
		email_qs = User.objects.filter(email=email)
		if email_qs.exists():
			raise forms.ValidationError("Este correo ya está en uso")
		return email