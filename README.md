# CV_stable
¡Claro! Para implementar Dreambooth sobre Stable Diffusion, sigue este tutorial paso a paso. Dreambooth permite la personalización de un modelo de generación de imágenes, con solo unos pocos ejemplos de lo que deseas entrenar.

### **Pasos a seguir:**

#### 1. **Configuración del entorno**

* **Instalar el entorno de `Textual Inversion`**: Primero, asegúrate de tener configurado el entorno adecuado para ejecutar el modelo. Si ya tienes instalado el entorno para `Textual Inversion`, puedes seguir este paso; de lo contrario, sigue las instrucciones del repositorio de **Textual Inversion** o **Stable Diffusion** para configurarlo.

* **Instalar dependencias**: Si estás utilizando `Textual Inversion`, probablemente necesitarás las dependencias de **PyTorch**, **CUDA**, y bibliotecas de difusión como `diffusers`.

```bash
pip install torch torchvision torchaudio
pip install transformers
```

#### 2. **Descargar el modelo pre-entrenado de Stable Diffusion**

* Dirígete a **HuggingFace** y descarga el archivo `sd-v1-4-full-ema.ckpt`. Este es el modelo preentrenado que necesitarás para la fase de entrenamiento.

```bash
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
```

Guarda el archivo en una ubicación conveniente.

#### 3. **Generación de imágenes para regularización**

* Para el algoritmo de Dreambooth, necesitas generar imágenes de regularización. Estas imágenes deben describir la clase de objeto que estás entrenando. Si estás entrenando para una clase como "perro", puedes usar el siguiente comando para generar imágenes.

```bash
python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt /path/to/sd-v1-4-full-ema.ckpt --prompt "a photo of a dog"
```

Este comando generará 8 imágenes de un perro, lo cual es suficiente, pero puedes generar más imágenes para mejorar la regularización, en especial si los objetos que deseas generar son complejos.

#### 4. **Entrenamiento del modelo**

* Con las imágenes generadas y las imágenes de entrenamiento listas, es momento de entrenar el modelo. Usa el siguiente comando para comenzar el entrenamiento. Asegúrate de haber configurado correctamente el archivo `v1-finetune_unfrozen.yaml`.

```bash
python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml \
               -t \
               --actual_resume /path/to/sd-v1-4-full-ema.ckpt \
               -n <nombre_del_trabajo> \
               --gpus 0, \
               --data_root /ruta/a/las/imagenes_de_entrenamiento \
               --reg_data_root /ruta/a/las/imagenes_de_regularizacion \
               --class_word perro
```

Aquí:

* `<nombre_del_trabajo>` es el nombre que le des al trabajo de entrenamiento.
* `--gpus 0,` indica que utilizarás la GPU 0. Si usas múltiples GPUs, ajusta esto.
* `--data_root` debe ser la ruta a las imágenes de entrenamiento personalizadas.
* `--reg_data_root` debe ser la ruta a las imágenes de regularización.
* `--class_word` es la clase que estás entrenando (por ejemplo, "perro").

**Configuración importante**:

* En el archivo `v1-finetune_unfrozen.yaml`, el parámetro `learning rate` se ajusta a 1e-6.
* El identificador especial se define como `sks` en este caso (puedes cambiarlo si lo prefieres).

#### 5. **Generación de imágenes personalizadas**

* Una vez que el modelo esté entrenado, puedes generar imágenes personalizadas usando el siguiente comando:

```bash
python scripts/stable_txt2img.py --ddim_eta 0.0 \
                                 --n_samples 8 \
                                 --n_iter 1 \
                                 --scale 10.0 \
                                 --ddim_steps 100 \
                                 --ckpt /ruta/a/tu/modelo_entrenado.ckpt \
                                 --prompt "photo of a sks dog"
```

Aquí, el `sks` es el identificador que se usó durante el entrenamiento. Si lo cambiaste, sustitúyelo por tu nuevo identificador.

#### 6. **Monitoreo del proceso de entrenamiento**

Durante el entrenamiento, el modelo guardará puntos de control en dos momentos: en el paso 500 y al final del entrenamiento. Los archivos se guardarán en la carpeta `./logs/<job_name>/checkpoints`. Puedes utilizar el checkpoint del paso 500 si deseas resultados más rápidos.

```bash
./logs/<job_name>/checkpoints/step_500.ckpt
```

**Tiempo de entrenamiento**:

* El entrenamiento de 800 pasos toma alrededor de 15 minutos en GPUs de alto rendimiento como A6000.

---

### **Consejos adicionales**:

* Si las imágenes generadas son de baja calidad o poco realistas, ajusta el número de imágenes de regularización, como se sugiere en el repositorio (100 o 200 imágenes pueden mejorar los resultados).
* Si usas un identificador muy común (como `man` o `woman`), puede ser útil encontrar imágenes diversas de estas clases para mejorar la regularización.

Con estos pasos, habrás personalizado tu modelo de Stable Diffusion usando Dreambooth para generar imágenes ajustadas a tus necesidades.
