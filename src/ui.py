from tkinter import filedialog, Tk, Label, Button, Toplevel, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
from processimage import process_image, predict
from skull_strip import stripSkull
from segment import segment_tumor, apply_intensity_color_map

# Function to save all processed images to a chosen directory
def save_all_images(images):
    folder_path = filedialog.askdirectory()  # Opens a dialog for user to select a folder
    if folder_path:
        for name, image in images:
            image.save(os.path.join(folder_path, name))  # Save each image with its name in the selected folder
        messagebox.showinfo("Export Successful", "All images have been exported successfully.")

# Function to upload and process images
def upload_action():
    global labels
    file_path = filedialog.askopenfilename()  # Open a dialog to select an image file
    if file_path:
        original_img = Image.open(file_path)
        original_img_resized = original_img.resize((250, 250))
        original_img_tk = ImageTk.PhotoImage(original_img_resized)

        labels_text = ["Original Image", "Skull-Stripped MRI", "Segmented MRI", "Color Mapped MRI"]
        for i, text in enumerate(labels_text):
            labels[i]['text'] = text  # Set label text

        # Display the original image
        if hasattr(upload_action, 'panel_original'):
            upload_action.panel_original.configure(image=original_img_tk)
        else:
            upload_action.panel_original = ttk.Label(root, image=original_img_tk)
            upload_action.panel_original.grid(column=0, row=2)
        upload_action.panel_original.image = original_img_tk

        # Process the image to remove the skull
        skull_stripped_img = stripSkull(file_path)
        skull_stripped_img_gray = cv2.cvtColor(skull_stripped_img, cv2.COLOR_BGR2GRAY)
        skull_stripped_img_pil = Image.fromarray(skull_stripped_img_gray).resize((250, 250))
        skull_stripped_img_tk = ImageTk.PhotoImage(skull_stripped_img_pil)

        # Display the skull-stripped image
        if hasattr(upload_action, 'panel_skull_stripped'):
            upload_action.panel_skull_stripped.configure(image=skull_stripped_img_tk)
        else:
            upload_action.panel_skull_stripped = ttk.Label(root, image=skull_stripped_img_tk)
            upload_action.panel_skull_stripped.grid(column=1, row=2)
        upload_action.panel_skull_stripped.image = skull_stripped_img_tk

        # Predict tumor presence
        processed_image = process_image(file_path)
        prediction = predict(processed_image)
        result_text = "No Tumor Detected" if prediction == 0 else "Tumor Detected"
        result_label.config(text=result_text)

        images_to_export = [("original.png", original_img_resized), ("skull_stripped.png", skull_stripped_img_pil)]

        # Further processing if tumor is detected
        if prediction != 0:
            segmented_image = segment_tumor(skull_stripped_img_gray)
            segmented_image_pil = Image.fromarray(segmented_image).convert("RGB").resize((250, 250))
            segmented_img_tk = ImageTk.PhotoImage(segmented_image_pil)
            images_to_export.append(("segmented.png", segmented_image_pil))

            # Display the segmented image
            if hasattr(upload_action, 'panel_segmented'):
                upload_action.panel_segmented.configure(image=segmented_img_tk)
            else:
                upload_action.panel_segmented = ttk.Label(root, image=segmented_img_tk)
                upload_action.panel_segmented.grid(column=2, row=2)
            upload_action.panel_segmented.image = segmented_img_tk

            # Apply color mapping to the image
            color_mapped_image = apply_intensity_color_map(skull_stripped_img_gray)
            color_mapped_image_pil = Image.fromarray(color_mapped_image).convert("RGB").resize((250, 250))
            color_mapped_img_tk = ImageTk.PhotoImage(color_mapped_image_pil)
            images_to_export.append(("color_mapped.png", color_mapped_image_pil))

            # Display the color mapped image
            if hasattr(upload_action, 'panel_color_mapped'):
                upload_action.panel_color_mapped.configure(image=color_mapped_img_tk)
            else:
                upload_action.panel_color_mapped = ttk.Label(root, image=color_mapped_img_tk)
                upload_action.panel_color_mapped.grid(column=3, row=2)
            upload_action.panel_color_mapped.image = color_mapped_img_tk

            export_button.config(state='normal', command=lambda: save_all_images(images_to_export))
        else:
            export_button.config(state='disabled')  # Disable export button if no tumor detected

root = Tk()
root.title("Tumor Detection")
root.geometry('1000x400')  # Set the size of the window

# GUI components setup
upload_button = ttk.Button(root, text="Upload Image", command=upload_action)  # Upload button
upload_button.grid(column=0, row=0)
export_button = ttk.Button(root, text="Export All Images", state='disabled')  # Export button
export_button.grid(column=3, row=0)
result_label = ttk.Label(root, text="", font=("Helvetica", 16))  # Label for displaying results
result_label.grid(column=1, row=3)

# Labels for each image type, initially set to empty string ("")
labels_text = ["", "", "", ""]
labels = []
for i, text in enumerate(labels_text):
    label = ttk.Label(root, text=text)
    label.grid(column=i, row=1)
    labels.append(label)  # Save a reference to the label

root.mainloop()  # Start the GUI application
