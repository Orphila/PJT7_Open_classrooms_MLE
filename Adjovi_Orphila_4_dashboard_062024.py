import streamlit as st
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

# Charger le modèle et le processeur
model_path = "./models"
#model = ViTForImageClassification.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained("Orphila/pjt7_model")

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)

val_transforms = Compose(
    [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ]
)

dataset = load_dataset('Orphila/pjt7_data')
dataset = dataset['test']
# Convertir le dataset en DataFrame
df = pd.DataFrame(dataset)


# Prétraitement de l'image
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = val_transforms(image)
    return image.unsqueeze(0)  # Ajouter une dimension pour le batch

# Faire une prédiction
def predict_dog_breed(image):
    inputs = preprocess_image(image)

    with torch.no_grad():
        outputs = model(pixel_values=inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    return predicted_class_idx

id_labels = {0: 'beagle',
             1: 'English_foxhound',
             2: 'Scottish_deerhound',
             3: 'Staffordshire_bullterrier',
             4: 'American_Staffordshire_terrier',
             5: 'Yorkshire_terrier',
             6: 'Lakeland_terrier',
             7: 'Airedale',
             8: 'flat-coated_retriever',
             9: 'German_short-haired_pointer'}
df['label_2'] = df['label'].map(id_labels)
############### Streamlit ###############
st.set_page_config(layout="wide")
st.title("Prédiction de race de chien")

st.header("Voici à quoi ressemblent les images contenues dans le stanford dataset")

classes_to_display= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Classes à afficher
samples_per_class = 1  # Nombre d'échantillons par classe à afficher

samples = []

for class_label in classes_to_display:
    # Filtrer les échantillons par classe
    samples_class = df[df['label'] == class_label].head(samples_per_class)

    samples.extend(samples_class.to_dict(orient='records'))

num_images_to_display = len(classes_to_display)
fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))    
for i in range (len(samples)) :
    sample = samples[i]
    image,label = sample['image'], sample['label']
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
st.pyplot(fig)
col1,col2 = st.columns(2)

with col1: 

    # Calculer les informations demandées
    class_counts = df['label_2'].value_counts()
    num_classes = len(class_counts)
    mean_images_per_class = class_counts.mean()
    min_images_per_class = class_counts.min()
    max_images_per_class = class_counts.max()
    example_class_names = class_counts.index[:10].tolist()

    # Créer un tableau avec les résultats
    table_data = {
        'Metric': ['Nombre de classes', 'Nombre moyen d\'images par classe', 'Nombre minimum d\'images par classe', 'Nombre maximum d\'images par classe', 'Exemples de noms de classes'],
        'Value': [num_classes, mean_images_per_class, min_images_per_class, max_images_per_class, ', '.join(example_class_names)]
    }

    table_df = pd.DataFrame(table_data)

    classes_to_display = ["beagle", 
                          "English_foxhound", 
                          "Scottish_deerhound", 
                          "Staffordshire_bullterrier", 
                          "American_Staffordshire_terrier", 
                          "Yorkshire_terrier", 
                          "Lakeland_terrier", 
                          "Airedale", 
                          "flat-coated_retriever",
                          "German_short-haired_pointer"]
 
    # Afficher les résultats dans Streamlit
    st.write("Voici quelques statistiques sur le dataset des chiens du Stanford Dataset:")
    st.table(table_df)


with col2:
    st.header("Prédictions")
    uploaded_file = st.file_uploader("Téléchargez une image de chien", type=["jpg", "png"])

    if uploaded_file is not None:
        # Afficher l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        # Prédiction
        prediction = predict_dog_breed(image)

        # Résultat de la prédiction
        st.subheader("Résultat de la prédiction :")
        st.write("Il s'agit d'un ",id_labels[int(prediction)])


