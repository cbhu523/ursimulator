import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import gensim
import timm
import ast
import re
from joblib import load
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, StableDiffusionInpaintPipeline
from mmseg.apis import init_model, inference_model
from skopt import gp_minimize
from skopt.space import Categorical
import csv
from sklearn.metrics.pairwise import cosine_similarity

# Global settings
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_DEPTH = 1
NUM_SIMILAR_WORDS = 1
BASE_N_CALLS = 10

# Load models/resources
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(torch_device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(torch_device)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(torch_device)
inpaint = StableDiffusionInpaintPipeline.from_pretrained('runwayml/stable-diffusion-inpainting', revision="fp16", torch_dtype=torch.float16).to(torch_device)
resnet_model = timm.create_model("resnet50", pretrained=True).to(torch_device)
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.eval()
clf = load('../perception_model/svm_safe.joblib')

config_file = '../../../configs/ocrnet/ocrnet_hr48_4xb2-160k_cityscapes-512x1024.py'
checkpoint_file = '../../../checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')
torch.backends.cudnn.benchmark = True
image_cache = {}
word_scores = {}
# Global list to store words and their scores
global_word_rankings = {}

modification={'building':'house','garden':'garden','park':'park','wall':'wall'}

def sanitize_filename(filename):
    # Replace invalid file name characters with an underscore
    sanitized = re.sub(r'[\/:*?"<>|]', '_', filename)

    # Truncate the file name if it's too long
    max_length = 255  # Maximum length for most file systems
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Handle reserved file names for Windows systems
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                      "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    if sanitized.upper() in reserved_names:
        sanitized = "_" + sanitized

    return sanitized

def clear_caches():
    global image_cache, word_scores
    image_cache = {}
    word_scores = {}

def preprocess_image(image_input):
    if isinstance(image_input, str) and image_input in image_cache:
        return image_cache[image_input]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
    processed_image = transform(image)
    
    if isinstance(image_input, str):
        image_cache[image_input] = processed_image
        
    return processed_image

def predict(image_path):
    input_tensor = preprocess_image(image_path).unsqueeze(0).to(torch_device)
    with torch.no_grad():
        features = resnet_model(input_tensor)
    features = features.squeeze(-1).squeeze(-1).cpu().numpy()
    prediction = clf.predict(features)
    return prediction[0]

def calculate_reward(original_score, edited_score):
    delta = edited_score - original_score
    return np.sign(delta) * np.sqrt(abs(delta)) 


def get_closest_word(vec):
    """Get the closest word in the word2vec_model vocabulary to a given vector."""
    all_words = word2vec_model.index_to_key  # List of words in the vocabulary
    similarities = cosine_similarity([vec], word2vec_model.vectors)
    return all_words[np.argmax(similarities)]


def objective_vectorized(direction, current_vector, img_path, binarized_mask, original_score, word_scores,target_object,output_csv_path,output_img):
    """Objective function operating in the word2vec space."""
    # Move in the given direction from the current vector
    new_vector = current_vector + direction
    # Convert the resultant vector to the nearest word
    word_str = get_closest_word(new_vector)
    
    _, reward, _ = evaluate_word(word_str, img_path, binarized_mask, original_score, word_scores,target_object,output_csv_path,output_img)
    return -reward

def optimize_word_embedding(img_path, binarized_mask, original_score, initial_word, word_scores, target_object, output_csv_path, output_img, depth=1):
    if depth > MAX_DEPTH:
        return None, -np.inf

    # Convert the initial word to its corresponding vector
    initial_vector = word2vec_model[initial_word]
    # Define the optimization space as directions in the word2vec space
    space = [(-1.0, 1.0) for _ in range(word2vec_model.vector_size)]
    
    n_calls_val = BASE_N_CALLS + len(space) // 2

    result = gp_minimize(
        lambda direction: objective_vectorized(direction, initial_vector, img_path, binarized_mask, original_score, word_scores,target_object,output_csv_path,output_img),
        space, n_calls=n_calls_val, random_state=0
    )
    
    # Get the best direction from the optimization result
    best_direction = result.x
    # Move in the best direction from the initial vector
    new_vector = initial_vector + best_direction
    # Convert this vector to the nearest word
    best_word = get_closest_word(new_vector)
    best_score = -result.fun
    
    # Recursive call for further refinement
    optimize_word_embedding(img_path, binarized_mask, original_score, best_word, word_scores, target_object, output_csv_path, output_img, depth=depth+1)

def evaluate_word(word, img_path, binarized_mask, original_score, word_cache,target_object,output_csv_path,output_img):
    global global_word_rankings  # Ensure we're using the global dictionary
    word = str(word)  # Convert word to native Python string
    if word in word_cache:
        edited_score = word_cache[word]
    else:
        prompt_para = word + ' ' + target_object + ' from street view'

        generator = torch.Generator(torch_device)
        im_result = inpaint(prompt=[prompt_para], image=Image.open(img_path), mask_image=binarized_mask, generator=generator).images[0]

        edited_score = predict(im_result)
        if edited_score > original_score:
            updated_word = sanitize_filename(word)
            edited_image_path = os.path.join(output_img,(updated_word+'.png'))
            # Create the output directory if it doesn't exist
            if not os.path.exists(output_img):
                os.makedirs(output_img)
            im_result.save(edited_image_path, 'PNG')  # Save the edited image

        word_cache[word] = edited_score
    reward = calculate_reward(original_score, edited_score)
    
    # Add the word, its score, and its reward to the global rankings if not already present
    if word not in global_word_rankings:
        global_word_rankings[word] = (edited_score, reward)
        # Write the results to the CSV file
    with open(output_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([word, edited_score, reward])
    return word, reward, edited_score

def process_image_bayesian_optimization(full_img_path,target,source,output_csv_path,output_img):
    
    # Reset caches and rankings for each image
    global_word_rankings = {}
    image_cache = {}
    word_scores = {}
    
    target_object = modification[target]
    source_seg = ast.literal_eval(source)
    
    original_score = predict(full_img_path)
    if full_img_path not in image_cache:
        image_cache[full_img_path] = Image.open(full_img_path)
    result = inference_model(model, full_img_path)
    data = result.pred_sem_seg.data.cpu().numpy()
    
    # Initialize an empty mask
    combined_mask = np.zeros_like(data[0], dtype=np.uint8)

    # Create masks for each value in x and combine them
    for value in source_seg:
    
        mask = np.where(data[0] == value, 1, 0).astype(np.uint8)  # Using 1 instead of 255 here
       
        combined_mask = np.logical_or(combined_mask, mask)
   
    # Convert the logical_or result back to uint8 and scale up to 255
    combined_mask = (combined_mask * 255).astype(np.uint8)
        
    # Convert the combined_mask to a PIL Image with mode 'L' (grayscale)
    binarized_mask = Image.fromarray(combined_mask)

    initial_word = "beautiful"  # This can be any word that represents a starting point in the embedding space

    optimize_word_embedding(full_img_path, binarized_mask, original_score, initial_word, word_scores,target_object,output_csv_path,output_img)
    
    clear_caches()  # Clear caches after processing each image

def main(directory,input_file_path):
    #image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Open the input and output CSV files
    output_csv_dir = './rankings'
    output_img_dir = './ranking_imgs'
    with open(input_file_path, 'r') as input_file:
    	csv_reader = csv.reader(input_file)
    	
    	# Write the header row to the output file (assuming the first row contains headers)
    	header = next(csv_reader)
    
    	# Iterate through each row in the input file
    	for row in csv_reader:
                
    	    full_img_path = os.path.join(directory, row[0])
    	    output_csv_path = os.path.join(output_csv_dir, row[0]+'_rankings.csv')
    	    output_img = os.path.join(output_img_dir,row[0].split('.JPG')[0])

    	    process_image_bayesian_optimization(full_img_path,row[1],row[2],output_csv_path,output_img)

if __name__ == "__main__":
    directory_path = './all_images'
    csv_file = './final_input_csv_exps.csv'
    main(directory_path,csv_file)
