from django.shortcuts import redirect, render
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .models import *
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.db.models import Q
from datetime import datetime
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import base64
import io
from pathlib import Path
# Create your views here.

BASE_DIR = Path(__file__).resolve().parent.parent

# Define your classes_dict and YogaClassifier class as in your prediction code
classes_dict = {0: 'Adho Mukha Svanasana', 1: 'Adho Mukha Vrksasana', 2: 'Alanasana', 3: 'Anjaneyasana', 4: 'Ardha Chandrasana', 5: 'Ardha Matsyendrasana', 6: 'Ardha Navasana', 7: 'Ardha Pincha Mayurasana', 8: 'Ashta Chandrasana', 9: 'Baddha Konasana', 10: 'Bakasana', 11: 'Balasana', 12: 'Bitilasana', 13: 'Camatkarasana', 14: 'Dhanurasana', 15: 'Eka Pada Rajakapotasana', 16: 'Garudasana', 17: 'Halasana', 18: 'Hanumanasana', 19: 'Malasana', 20: 'Marjaryasana', 21: 'Navasana', 22: 'Padmasana', 23: 'Parsva Virabhadrasana', 24: 'Parsvottanasana', 25: 'Paschimottanasana', 26: 'Phalakasana', 27: 'Pincha Mayurasana', 28: 'Salamba Bhujangasana', 29: 'Salamba Sarvangasana', 30: 'Setu Bandha Sarvangasana', 31: 'Sivasana', 32: 'Supta Kapotasana', 33: 'Trikonasana', 34: 'Upavistha Konasana', 35: 'Urdhva Dhanurasana', 36: 'Urdhva Mukha Svsnssana', 37: 'Ustrasana', 38: 'Utkatasana', 39: 'Uttanasana', 40: 'Utthita Hasta Padangusthasana', 41: 'Utthita Parsvakonasana', 42: 'Vasisthasana', 43: 'Virabhadrasana One', 44: 'Virabhadrasana Three', 45: 'Virabhadrasana Two', 46: 'Vrksasana'}

class YogaClassifier(torch.nn.Module):
    # Your implementation here
    def __init__(self, num_classes, input_length):
        super(YogaClassifier, self).__init__()
        self.layer1 = torch.nn.Linear(input_length, 64)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.layer2 = torch.nn.Linear(64, 64)
        self.outlayer = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.outlayer(x)
        return x
    pass

def load_model():
    # Your implementation here
    model_pose = YogaClassifier(num_classes=len(classes_dict), input_length=32)
    model_pose.load_state_dict(torch.load("best.pth"))
    model_pose.eval()
    return model_pose
    pass


def make_prediction(model, image_path):
    model_yolo = YOLO("yolov8x-pose-p6.pt")

    results = model_yolo.predict(image_path, verbose=False)
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        keypoints = r.keypoints.xyn.cpu().numpy()[0]
        keypoints = keypoints.reshape((1, keypoints.shape[0] * keypoints.shape[1]))[0].tolist()

        # Preprocess keypoints data
        keypoints_tensor = torch.tensor(keypoints[2:], dtype=torch.float32).unsqueeze(0)

        # Prediction
        model.cpu()
        model.eval()
        with torch.no_grad():
            logit = model(keypoints_tensor)
            pred = torch.softmax(logit, dim=1).argmax(dim=1).item()
            prediction = classes_dict[pred]

        # Convert the plot image to base64 string
        image = io.BytesIO()
        plt.imshow(im_array[..., ::-1])
        plt.title(f"Prediction: {prediction}", color="green")
        plt.savefig(image, format='png')
        plt.close()
        image.seek(0)
        plot_base64 = base64.b64encode(image.read()).decode('utf-8')
        return plot_base64, prediction



def index(request):
    return render(request, 'index.html')


def login(request):

    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']
        user = authenticate(username=email, password=password)
        if user is not None:
            if user.is_active:
                if user.is_superuser:
                    return redirect("adminHome")
                elif user.is_staff:
                    if user.last_name == "Doctor":
                        shop = Doctor.objects.get(email=email)
                        request.session['id'] = shop.id
                        return redirect("Doc_home")
                    elif user.last_name == "Trainer":
                        shop = Trainer.objects.get(email=email)
                        request.session['id'] = shop.id
                        return redirect("Trainer_home")
                else:
                    cust = Customer.objects.get(email=email)
                    request.session['id'] = cust.id
                    return redirect("custHome")
            else:
                messages.info(request, 'Account is not active..')
                return render(request, 'login.html')
        else:
            messages.info(request, 'Please check the email..')
            return render(request, 'login.html')
    else:
        return render(request, 'login.html')


def custHome(request):
    if request.POST:
        try:
            # Get the image file from the request
            image_file = request.FILES['file']
            # Save the image file to a temporary location
            # image_path = 'temp.png'
            # image_file.save(image_path)
            toTable = Yoga_Images.objects.create(img = image_file)
            toTable.save()
            image_path_row = Yoga_Images.objects.latest('id')
            image_path = f"{BASE_DIR}\\static\\media\\{image_path_row.img}"

            # Load the model
            model = load_model()
            # Make prediction
            plot_base64, prediction = make_prediction(model, image_path)

            # Remove the temporary image file
            os.remove(image_path)

            #prosdict
            prosdict = { 'Adho Mukha Svanasana':"improves strength, balance and flexibility. It calms the brain, Reduces stiffness in the shoulder region and tones the legs.",  'Adho Mukha Vrksasana':"improves strength, balance and flexibility.",  'Alanasana':"improves strength, balance and flexibility.",  'Anjaneyasana':"improves strength, balance and flexibility.",  'Ardha Chandrasana':"improves strength, balance and flexibility.",  'Ardha Matsyendrasana':"improves strength, balance and flexibility.",  'Ardha Navasana':"improves strength, balance and flexibility.",  'Ardha Pincha Mayurasana':"improves strength, balance and flexibility.",  'Ashta Chandrasana':"improves strength, balance and flexibility.",  'Baddha Konasana':"improves strength, balance and flexibility.", 'Bakasana':"improves strength, balance and flexibility.",  'Balasana':"improves strength, balance and flexibility. It makes the spine flexible and broadens the chest, Makes the neck, shoulders, chest and head more active, Increase blood circulation, Gives power and flexibility to the body, It reduces obesity, Helpful in strengthening the digestive system and increasing the lung power",  'Bitilasana':"improves strength, balance and flexibility.",  'Camatkarasana':"improves strength, balance and flexibility.",  'Dhanurasana':"improves strength, balance and flexibility. It makes the spine flexible and reduces its stiffness, Reduces obesity, It can cure stomach pains, Strengthens the muscles of the arms, legs and stomach, It increases the lung power and breathing process.",  'Eka Pada Rajakapotasana':"improves strength, balance and flexibility.",  'Garudasana':"improves strength, balance and flexibility.",  'Halasana':"improves strength, balance and flexibility. Helpful in strengthening the neck muscles, Helpful in reducing weight and back pain, Strengthens the backbone Improves blood circulation.",  'Hanumanasana':"improves strength, balance and flexibility.",  'Malasana':"improves strength, balance and flexibility. Strengthens the Lower Back, Helps in stretching the groin and also the lower back, Tones the belly, Releases tension in the hips and knees.",  'Marjaryasana':"improves strength, balance and flexibility.",  'Navasana':"improves strength, balance and flexibility.",  'Padmasana':"improves strength, balance and flexibility.",  'Parsva Virabhadrasana':"improves strength, balance and flexibility.",  'Parsvottanasana':"improves strength, balance and flexibility.",  'Paschimottanasana':"improves strength, balance and flexibility.",  'Phalakasana':"improves strength, balance and flexibility.",  'Pincha Mayurasana':"improves strength, balance and flexibility.",  'Salamba Bhujangasana':"improves strength, balance and flexibility. It makes the spine flexible and broadens the chest, Makes the neck, shoulders, chest and head more active, Increase blood circulation, Gives power and flexibility to the body, It reduces obesity, Helpful in strengthening the digestive system and increasing the lung power",  'Salamba Sarvangasana':"improves strength, balance and flexibility.",  'Setu Bandha Sarvangasana':"improves strength, balance and flexibility.",  'Sivasana':"improves strength, balance and flexibility.",  'Supta Kapotasana':"improves strength, balance and flexibility.",  'Trikonasana':"improves strength, balance and flexibility. Improves flexibility of the spine and relieves back pain and stiffness in the neck area, Massages and tones the pelvic region, relieves gastritis, indigestion and acidity, Helps you to improve your posture too.",  'Upavistha Konasana':"improves strength, balance and flexibility.",  'Urdhva Dhanurasana':"improves strength, balance and flexibility.",  'Urdhva Mukha Svsnssana':"improves strength, balance and flexibility.",  'Ustrasana':"improves strength, balance and flexibility.",  'Utkatasana':"improves strength, balance and flexibility.",  'Uttanasana':"improves strength, balance and flexibility.",  'Utthita Hasta Padangusthasana':"improves strength, balance and flexibility.",  'Utthita Parsvakonasana':"improves strength, balance and flexibility.",  'Vasisthasana':"improves strength, balance and flexibility.",  'Virabhadrasana One':"improves strength, balance and flexibility.",  'Virabhadrasana Three':"improves strength, balance and flexibility.",  'Virabhadrasana Two':"improves strength, balance and flexibility.",  'Vrksasana':"improves strength, balance and flexibility."}

            # Return the Prediction result and plot to the prediction.html template
            pros = prosdict[str(prediction)]

            return render(request,'prediction.html', {"prediction":prediction, "plot_base64":plot_base64,"pros":pros})
        except:
            messages.info(request, 'Some error occured, please input a valid image')
            return redirect("/custHome")
    return render(request, "custHome.html")




def logout(request):
    if 'id' in request.session:
        request.session.flush()
    return redirect('login')

def dietChart(request):
    return render(request, 'dietChart.html')
