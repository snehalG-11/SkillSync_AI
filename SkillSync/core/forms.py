from django import forms
from django.contrib.auth.models import User
class ResumeUploadForm(forms.Form):
    resume = forms.FileField(
        widget=forms.ClearableFileInput(attrs={'accept': '.pdf,.doc,.docx,.txt'})
    )

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get("password")
        p2 = cleaned_data.get("confirm_password")

        if p1 and p2 and p1 != p2:
            raise forms.ValidationError("Passwords do not match")

        return cleaned_data