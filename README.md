
# MediaPipe y REBA

> [!TIP]
> Puedes decargar PyCharm en este link pagina oficial: https://www.jetbrains.com/es-es/pycharm/download/?section=windows
> 
> Tambien puedes buscar PyCharm en tu navegador de preferencia.

> [!IMPORTANT]
> Version de Python requerida 3.11

<P> Al crear el nuevo proyecto en Pycharm el nos deja selecionar la version de Python que deseamos utilizar.</P>


<img src="https://github.com/user-attachments/assets/f46d2954-8c28-49b5-9680-dc447a730dbe" alt="imagen">

<p>Tambien podemos decargar la version correspondiente de Python por nuestro navegador web escribiendo Python y ingresando a https://www.python.org/ </p>

 <img src="https://github.com/user-attachments/assets/02f6f022-921d-42c9-9c61-659d2c55939d" alt="imagen">

<p>Si estas utilizando PyCharm puedes configurarlo directamente en el proyecto entrando a Configuracion y Python Interpeter</p>


 <img src="https://github.com/user-attachments/assets/73a0e433-05e6-4f84-a40f-9d0f80820a0a" alt="imagen">


<p> En este repositorio vamos a  encontrar 3 archivos .py dos de los cuales son unicamente Mediapipe uno para imagenes otro para video y uno para mediapipe y calculo de reba</p>

1. Instacia_REBA.py
2. mediapipe_video.py
3. mediapipe_imagen.py


# mediapipe_imagen.py

<p> Ingresando al archivo mediapipe_imagen.py encontraremos el siguiente codigo primeramente importaremos el CV2 y el mediapipe si no lo tienen instaldo y estan usando PyCharm al ejecutar el codigo en la parte inferior mostrara "Install packages" el cual les realiza la instalacion correspondiente del Mediapipe esto se visualiza al ejecutar el codigo o ejecuando el comando en la terminal "pip install mediapipe".</p>


 ```
   pip install mediapipe
 ```

<img src="https://github.com/user-attachments/assets/848e0176-188e-4663-90ff-204f985b4355" alt="imagen">


<p>Esta son las lineas de codigo presentes la linea que contiene "image = cv2.imread("prueba2.jpg")"  es donde podemos realizar el cambio de archivo correspondiente al que deseamos trabajar en este caso se llama prueba2.jpg</p>

 ```
  import cv2
  import mediapipe as mp
  
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  
  with mp_pose.Pose(
          static_image_mode=True) as pose:
      image = cv2.imread("prueba2.jpg")
      height, width, _ = image.shape
      image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
      results = pose.process(image_rgb)
      print("Pose landmarks:", results.pose_landmarks)
  
      if results.pose_landmarks is not None:
          mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS)
  
  cv2.imshow("Image", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

 ```


