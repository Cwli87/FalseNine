# Soccer Players > raw-images_Classes-Player_Ref_Ball-150imgs
https://universe.roboflow.com/roboflow-universe-projects/soccer-players-ckbru

Provided by a Roboflow user
License: CC BY 4.0

# Soccer (Fùtbol) Match Analysis
## FIFA World Cup - Qatar 2022
### United States of America (USA) vs. Netherlands (NED) on Dec. 3, 2022

This dataset includes classes for:
* `USA` (USA players)
* `NED` (NED players)
* `Ball` (soccer/fùtbol)
* `Ref` (referee)
* `Goalie`
* `GOAL` (soccer/fùtbol goal)

(video/GIF slowed to help with visualization)
![Example Video from Deploy Tab](https://i.imgur.com/PLS0HB3.gif)

### How to Use this Dataset/Project:
This is a great starter-dataset for anyone wanting to make computer vision models for player-tracking, ball-tracking, or other game/match analytics.

* If the player-classes are left as `USA` and `NED`, then you will be tracking players with labels of `USA` and `NED`.
* If the player-classes are updated to `Team1` and `Team2` with [Modify Classes](https://docs.roboflow.com/image-transformations/image-preprocessing#modify-classes), then you will be tracking players with labels of `Team1` and `Team2`.
* If the player-classes are *all* updated to `Player` with [Modify Classes](https://docs.roboflow.com/image-transformations/image-preprocessing#modify-classes), then you will be tracking all players on the pitch with labels of `Player`.

### Next Steps/Dataset and Model Improvement:
Try [Cloning this project](https://help.roboflow.com/dataset-upload-cloning-images-from-roboflow-universe) and adding more of [your own images or video](https://docs.roboflow.com/adding-data); images from [another project on Roboflow Universe](https://universe.roboflow.com/), or a [video from YouTube](https://blog.roboflow.com/youtube-video-computer-vision/).
* Roboflow Universe: [All Categories](https://universe.roboflow.com/browse/) | [Sports](https://universe.roboflow.com/browse/sports)
* Try [Active Learning](https://blog.roboflow.com/what-is-active-learning/) to root out false and low-confidence detections to make a more highly-performant model!
	* https://docs.roboflow.com/python/active-learning


UPDATES (12/15/2022):
* More [image data (video frames) from YouTube](https://blog.roboflow.com/youtube-video-computer-vision/) was added from the [France vs. Morocco 2022 World Cup Semi-Final Match](https://www.youtube.com/watch?v=aZ0wo2_1iVg) to help make the model for generalizable for other soccer/fùtbol matches.