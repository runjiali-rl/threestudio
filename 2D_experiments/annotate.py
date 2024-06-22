import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate images with sentences.")
    parser.add_argument("--folder_path", type=str, default="2D_experiments/generated_images/composite", help="Path to the folder containing images.")
    parser.add_argument("--output_dir", type=str, default="2D_experiments/results/composite", help="Path to the output CSV file.")
    return parser.parse_args()

class ImageSentenceApp:
    def __init__(self, root, images_and_sentences, output_csv):
        self.root = root
        self.root.title("Image and Sentence Verification")
        self.current_index = 0
        self.output_csv = output_csv

        self.images_and_sentences = images_and_sentences
        self.results = []

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.sentence_label = tk.Label(root, text="", font=("Arial", 16))
        self.sentence_label.pack()

        self.yes_button = tk.Button(root, text="Yes", command=self.yes_action)
        self.yes_button.pack(side=tk.LEFT, padx=20)

        self.no_button = tk.Button(root, text="No", command=self.no_action)
        self.no_button.pack(side=tk.RIGHT, padx=20)

        self.load_image_and_sentence()

    def load_image_and_sentence(self):
        if self.current_index < len(self.images_and_sentences):
            image_path, sentence = self.images_and_sentences[self.current_index]
            self.image = Image.open(image_path)
            self.image = self.image.resize((400, 300))
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.sentence_label.config(text=sentence)
        else:
            self.save_results()
            messagebox.showinfo("End", "No more images and sentences.")
            self.root.quit()

    def yes_action(self):
        self.log_result("Yes")
        self.next_image()

    def no_action(self):
        self.log_result("No")
        self.next_image()

    def log_result(self, result):
        image_path, sentence = self.images_and_sentences[self.current_index]
        self.results.append((image_path, sentence, result))

    def next_image(self):
        self.current_index += 1
        self.load_image_and_sentence()

    def save_results(self):
        with open(self.output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Path", "Sentence", "Result"])
            writer.writerows(self.results)


def load_images_and_sentences(folder_path):
    all_model_images_and_sentences = {}
    model_names = os.listdir(folder_path)
    for model_name in model_names:
        if model_name not in all_model_images_and_sentences:
            all_model_images_and_sentences[model_name] = []
        model_path = os.path.join(folder_path, model_name)
        for filename in os.listdir(model_path):
            if filename.endswith(".png"):
                image_path = os.path.join(model_path, filename)
                sentence = filename.split(".")[0].replace("_", " ")
                all_model_images_and_sentences[model_name].append((image_path, sentence))
            
    return all_model_images_and_sentences



def main():
    args = parse_args()
    folder_path = args.folder_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    all_model_images_and_sentences = load_images_and_sentences(folder_path)
    for model_name, images_and_sentences in all_model_images_and_sentences.items():
        print(f"Model: {model_name}")
        output_csv = os.path.join(output_dir, f"{model_name}.csv")
        if os.path.exists(output_csv):
            print(f"Skipping {model_name}.csv as it already exists.")
            continue
        root = tk.Tk()
        app = ImageSentenceApp(root, images_and_sentences, output_csv)
        root.mainloop()
        root.destroy()



if __name__ == "__main__":
    main()
