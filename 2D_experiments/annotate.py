import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# 示例图片和句子
images_and_sentences = [
    ("sunset.png", "This is a cat."),
    ("sunset.png", "This is a dog."),
    # 添加更多的图片和句子
]

class ImageSentenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image and Sentence Verification")
        self.current_index = 0

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
        if self.current_index < len(images_and_sentences):
            image_path, sentence = images_and_sentences[self.current_index]
            self.image = Image.open(image_path)
            self.image = self.image.resize((400, 300), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.photo)
            self.sentence_label.config(text=sentence)
        else:
            messagebox.showinfo("End", "No more images and sentences.")
            self.root.quit()

    def yes_action(self):
        self.next_image()

    def no_action(self):
        self.next_image()

    def next_image(self):
        self.current_index += 1
        self.load_image_and_sentence()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSentenceApp(root)
    root.mainloop()
