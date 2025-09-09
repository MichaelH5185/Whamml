# Damage Detection Project
## Getting Images
- I will upload the videos and the maps of the area they represent to [onedrive](https://mailmissouri-my.sharepoint.com/:f:/g/personal/mjhf89_umsystem_edu/EhGDOTgKdXhOtuxAkwNNXXsBui1cQnenn265BD2JKhCmAQ?e=iDKXkY) and send links out to the team.
- You can extract the images and then use a program like [pot player](https://potplayer.daum.net/) to extract images from the videos
- If you are using pot player, open the mp4 file and then press ctrl+G to open the consecutive image capture menu.
- Set the location you would like the images to go using the Storage line at the top.
- I suggest setting the rate of capture to something like 3000ms or 5000ms
- I would also recommend setting the file prefix to something link img
- Some of the videos start with wide angled views of the city; since this will lead to a large amount of unaccessible buildings please either crop the images or not use them.
## Uploading the Images to Roboflow
- I will add your emails to the Roboflow project when we start
- To upload your images, navigate to Animal_Project > Upload Data and either upload your folder with all your images or use upload files to upload individual images.
- This will add them to our annotation pool
## Annotating Images
- To annotate images, navigate to Animal_Project > Annotate.
- Select the first image to start.
- We are using polygonal bboxes so you can either use the polygon tool or the smart polygon tool to select a building in the image.
- To find the rating for the building, use the .kmz file in this repo and open it in either the [website](https://earth.google.com/web/) or the [desktop app](https://www.google.com/earth/about/versions/)
- Each building should have a dot with a rating on it.
- This repo also includes the raw data if you want to use that instead.
- Select the correct rating from the list and press enter
- Please label all buildings in the image, if they are too far in the background or blocked in some way please label it as 5-unaccessible in Roboflow
- Your annotations per image should preferably be less than 30 (if you excede it nothing bad will happen its just an efficiency thing)
## Adding your images to the dataset
- Navigate to Animal_Project > Annotate > Annotated and select Add # Images to Dataset  
If you have any questions you can send me an email @ mjhf89@umsystem.edu or a text @ 3142015915. Thank you for your Help! 
