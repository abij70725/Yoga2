# Generated by Django 5.0.3 on 2024-03-18 12:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rentalApp', '0003_chat_receiver'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='chat',
            name='receiver',
        ),
        migrations.AddField(
            model_name='chat',
            name='sender2',
            field=models.CharField(max_length=20, null=True),
        ),
    ]
