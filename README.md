# SDC-vehicle-detection
The goal of this project is to implement a simple vehicle detection pipeline from the vision system in the car. Vehicles are detected by first training a classifier on car and non-car sample data sets and using a sliding window search on the input frames to detect cars. The features vectors, heat map thresholds, sliding window scales used to train the classifier and detect car images have been experimented with quite a bit to derive at the final values.

Even though HOG classifier coupled with an SVM classifier is simple, it is clear that these techniques are not sufficient to build a robust pipeline to detect vehicles in real time. A hardware software hybrid approach where some of these image processing functions are implemented in hardware might be needed. 

Key steps of this project are:
* Examined the dataset of car and non-car images
* Extracted featureset for the images by appending 1) Spatial transform 2) Color histograms on all channels and 3) Histogram of Oriented Gradients (HOG) 
* Train a classifier Linear SVM classifier
* Implement a sliding window search on the input frames and use the trained classifier to search for vehicles in each window
* For all classifications, create a cumulative heatmap on multiple frames and apply a threshold to remove false positives. 
* Estimate a bounding box for vehicles detected

---
### Code

Run the python notebook `vehicle_detection.ipynb` for detecting vehicles in the images and video. Implementation consists of the following files located in the source directory

* source/vehicle_detection.ipynb  :   Runs the pipeline on test images and project video   
* source/tracking_pipeline.py     :   Implements functions required for feature collection, SVM classification and vehicle detection 
* out_images                      -   Folder with images at different stages of the pipeline
* out_videos                      -   Folder with lane detected output videos 

### Data Set

The data set used is comprised of images taken from the GTI vehicle image database, the KITTI vision benchmark suite. They comprise of two sets - car images and non-car images. Total data set contains

| Dataset       | Images        | 
|:-------------:|:-------------:| 
| Car           | 8792          | 
| Non-car       | 8968          |

Plotting some random samples from both the datasets

![Original](./writeup_images/sample_car_images.png)
![Original Image](./writeup_images/sample_noncar_images.png)




```python
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```

Then, we find internal corners of the chessboard using the Opencv function `cv2.findChessboardCorners()` and add the (x,y) coordinates to image space as shown below  
```python
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
```

Finally calibration matrix (mtx) and distortion coefficients (dst) are calculated using the `cv2.calibrateCamera()` function
```python
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```    

To remove distortion in an image, the function `cv2.undistort()` is applied with calibration matrix and distortion coefficients found above.
```python
    dst = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
```

Applying this on chessboard images, we get 

![Original Distorted Image](./writeup_images/camera_dist_correct.png)

We can clearly see distortion at the top of the left image corrected after applying `cv2.undistort()`

###Image Pipeline 
####1. Distortion correction

Applying the same distortion correction as above
![alt text](./writeup_images/dist_road.png)

####2. Binary thresholding using Gradient and Color tranforms 

A combination of color and gradient thresholds was used to generate the binary image. Four different thresholds were used to generate the thresholded binary image. 

* S-color tranform
* SobelX gradient
* Sobel gradient magnitude
* Sobel gradient direction

The following thresholds were narrowed based on experimentation.

| Transform               | Threshold     | 
|:-----------------------:|:-------------:| 
| S color                 | 170, 255      | 
| SobelX grad             | 20, 100       |
| Sobel gradmagnitude     | 20, 100       |
| Sobel graddirection     | 0.7, 1.3      |

The final thresholded image is obtained by combining the various transforms as shown below. The code for thresholding is implemented in the file `source/gen_process_image.py`

```python
    combined_binary[(s_binary == 1) | (sxbinary == 1) | ((smagbinary == 1) & (sdirbinary == 1))] = 1
```

The images below show the effect of thresholding. The top image shows SobelX gradient and Color transform apllied, whereas the bottom image shows the result with all four thresholds applied

![alt text](./writeup_images/gradient_threshold.png)

####3. Perspective transform

The thresholded image is then run through a Perspective tranform to generate a birds-eye view image. This is accomplished by the opencv functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()`

```python 
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
```

This source and destination points taken for the perspective transform are shown below.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 100, 0        | 
| 735, 460      | 1180, 0       |
| 0, 720        | 100, 720      |
| 1280, 720     | 1180, 0       |

As expected the source and destination points we pick impact the tranformed image quite a bit. This is more pronounced when the images contain shadows. An interesting observation is that occasionally better perspective transform and lane detection are achieved when the source images were taken to the ends of the image (rather than to the ends of the lane). 

Shown below are a thresholded image before and after the perspective transform is applied 

![alt text](./writeup_images/perspective.png)


####4. Identifying lane-lines and polyfit

The next step is to identify lane lines from the perspective trasformed image. For most instances, thresolding coupled with perspective transform provide reasonably clean outlines of the lane pixels. A sliding window technique is then used to identify the lane pixels. 

This section is implemented in `gen_lanefit.py`

First, a histogram of ON pixels is run the bottom half of image. 

```python
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
```

Then the location high intensity areas on the left and right sections of image are identified to give a starting location for the sliding window. 

```python
    end_margin_px = 100

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[end_margin_px:midpoint]) + end_margin_px
    rightx_base = np.argmax(histogram[midpoint+end_margin_px:histogram.shape[0]-100]) + midpoint + end_margin_px
```

The sliding window is moved along the the image and for each iteration of the window non-zero pixels in x and y direction are idenitifed.

```python
     good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                                (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
     good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
```

These good indices are appended to an array. At the end of each iteration, the mean of non-zero pixels is used to center the sliding windows of the next iteration. If there are not enough pixels, then the location of the window stays the same as before. 

```python
    if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
```

Once the sliding window is moved across the entire image, the non-zero x and y pixels are curve fitted using a 2nd order polynomial to detect lane lines  

```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

Shown below is the curve fitted lane lines with sliding windows and histogram of pixels  

![alt text](./writeup_images/curvefit.png)

Even in the limited test video provided, there are interesting cases where the entire thresholding and lane detection pipeline fails. They fall primarily in two areas
* Frames where the ends of the image do not have any active(ON) pixels since the line is dotted. Due to the nature of polyfit, this almost always returns an erroroneus fit
* Frames with shadows which make the processed images extremely noisy making it harder to even detect lane lines resulting in gross failures

Error correction for both these cases are implemented as shown below
In both these cases, the result is manifested as the right dotted white line detected being too far off (to the left or right) from its actual location. Here we measure the average road width and compare if it changed significantly (more than 15%) and apply correction  

First we measure the average roadwidth and curvature of the road as shown below

```python
    curr_road_width = np.average(right_fitx - left_fitx)    
    lc_rad, rc_rad = calc_curv(left_fitx, right_fitx, ploty)
```

If the detected roadwidth changes significantly compared to the previous frame, it is ignored. If the leftlane is calculated with good precision, then the right lane is calculated by just adding the average roadwidth to the left lane

```python
    if ((curr_road_width < 0.85*avg_road_width) | (curr_road_width > 1.15*avg_road_width) | (rc_rad < 50)):
         curr_road_width = avg_road_width
         right_fitx = left_fitx + curr_road_width
    else:
         avg_road_width = curr_road_width
```

####5. Metrics - Radius of curvature & Offset from center

Radius of curvature and vehicle offset from center is calculated in the file `gen_stats_display.py`

 First, the lanes detected in pixels are converted to lanes in real world meters and curve fitted 
```python
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
```    

and then radii of curvature are calculated based on the formula below

```python 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix 
                                + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix 
                                + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

Offset from center is calculated based on the assumption that the camera is the center of the image. 
```python
    xm_per_pix = 3.7/700
    
    offset_px = (center - 0.5*(leftx[y_eval] + rightx[y_eval]))   
    offset = xm_per_pix * offset_px
```
    
####6. Pipeline output
All the functions for polyfill `filled_image()` and anotation `anotate_image()` are included in the file `gen_stats_display.py`

First the detected lane is mapped on the warped image using the function `cv2.fillPoly()` and it is then converted into original image space using inverse perspective transform `cv2.warpPerspective()`

```python
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (orig_image.shape[1], orig_image.shape[0]))     
```

This entire pipeline is implemented in the file `gen_detection_pipeline.py`. Shown below is an image before and after passing through the pipeline

![alt text](./writeup_images/pipeline.png)
---

###Video Output

Here are links to the [video output](./output_video/adv_lane_track.mp4).

Another version is shown [here](./output_video/adv_lane_track1.mp4). The difference in both videos is mostly due to the areas selected for perspective transform and thresholds selected for color and gradient transforms. 

---

###Discussion and further work
This project is aN introduction to camera calibration, color and perspective transforms and curve fitting functions. However, it is not very robust and depends heavily on many factors going right. 

As you can see the pipeline is not robust in areas where the road has strong shadows and is wobbly. Also sections of the road with lighter color(concrete sections) combined with reflections of the sun make detecting lane especially the white dotted right lines much harder. There is already significant volume of academic research on shadow detection and elimination in images and this is an area that i would like understand and implement in the near future.
