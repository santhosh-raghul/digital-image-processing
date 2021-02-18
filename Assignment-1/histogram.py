# this program does histogram equalization and matching with rgb images
# code for grayscale is in the jupyter notebook 

import cv2,numpy as np,sys
from skimage import exposure
from matplotlib import pyplot as plt

def equalize_hist(img):
	
	freq=[]
	cum_freq=[]

	for i in range(img.shape[2]):
		freq.append(cv2.calcHist([img],[i],None,[256],[0,256]))
		cum_freq.append(np.zeros_like(freq[i]))
		cum_freq[i][0]=freq[i][0]

	freq=np.array(freq)

	for i in range(1,len(freq[0])):
		cum_freq[0][i]=cum_freq[0][i-1]+freq[0][i]
		cum_freq[1][i]=cum_freq[1][i-1]+freq[1][i]
		cum_freq[2][i]=cum_freq[2][i-1]+freq[2][i]

	sf=255.0/(img.shape[0]*img.shape[1])

	map=[[],[],[]]
	for i in range(256):
		map[0].append(np.round(cum_freq[0][i]*sf))
		map[1].append(np.round(cum_freq[1][i]*sf))
		map[2].append(np.round(cum_freq[2][i]*sf))
	
	equalized_img=np.zeros_like(img)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			equalized_img[i][j][0]=map[0][img[i][j][0]]
			equalized_img[i][j][1]=map[1][img[i][j][1]]
			equalized_img[i][j][2]=map[2][img[i][j][2]]

	# hist_b = cv2.calcHist([equalized_img],[0],None,[256],[0,256])
	# hist_g = cv2.calcHist([equalized_img],[1],None,[256],[0,256])
	# hist_r = cv2.calcHist([equalized_img],[2],None,[256],[0,256])

	# plot=plt.figure(filename)
	# plt.plot(hist_b,color='b')
	# plt.plot(hist_g,color='g')
	# plt.plot(hist_r,color='r')
	# plt.title(f'histogram for {filename}')
	# plt.show()

	return equalized_img.astype(np.uint8),np.array(map).astype(np.uint8)

def handle_missing_values(map_inv):
	while map_inv.count(-1)!=0:
		for i in range(256):
			if map_inv[i]==-1:
				c1 = i!=0 and map_inv[i-1]!=-1
				c2 = i!=255 and map_inv[i+1]!=-1
				if c1 and c2:
					map_inv[i]=(map_inv[i-1]+map_inv[i+1])/2
				elif c1:
					map_inv[i]=map_inv[i-1]
				elif c2:
					map_inv[i]=map_inv[i+1]

def match_hist(dest_img,source_img):

	eq_dest,map_dest=equalize_hist(dest_img)
	eq_source,map_source=equalize_hist(source_img)

	map_source_inv=[[-1 for i in range(256)] for j in range(3)]
	for i in range(256):
		map_source_inv[0][map_source[0][i][0]]=i
		map_source_inv[1][map_source[1][i][0]]=i
		map_source_inv[2][map_source[2][i][0]]=i

	handle_missing_values(map_source_inv[0])
	handle_missing_values(map_source_inv[1])
	handle_missing_values(map_source_inv[2])

	matched_img=np.zeros_like(dest_img)

	for i in range(dest_img.shape[0]):
		for j in range(dest_img.shape[1]):
			matched_img[i][j][0]=map_source_inv[0][eq_dest[i][j][0]]
			matched_img[i][j][1]=map_source_inv[1][eq_dest[i][j][1]]
			matched_img[i][j][2]=map_source_inv[2][eq_dest[i][j][2]]

	return matched_img

if __name__ == "__main__":

	argc=len(sys.argv)
	if argc not in [2,3]:
		print(f"correct usage:\npython3 {sys.argv[0]} image\n  > for histogram equalization (or)\npython3 {sys.argv[0]} dest_image source_image\n  > for histogram matching")
		sys.exit()

	elif(argc==2):

		img=cv2.imread(sys.argv[1])
		equalized_user_def,map=equalize_hist(img)
		cv2.imshow(f"equalized",equalized_user_def)
		cv2.waitKey(0)

		equalized_inbuilt=cv2.equalizeHist(img)
		cv2.imshow("inbuilt",equalized_inbuilt)
		cv2.waitKey(0)

		diff_img=np.abs(equalized_inbuilt.astype(np.int16)-equalized_user_def)
		cv2.imshow("difference",diff_img)
		cv2.waitKey(0)

	else:

		img1=cv2.imread(sys.argv[1])
		img2=cv2.imread(sys.argv[2])

		matched_user_def=match_hist(img1,img2)
		cv2.imshow("user defined",matched_user_def)
		cv2.waitKey(0)

		matched_inbuilt=exposure.match_histograms(img1,img2,multichannel=True).astype(np.uint8)
		cv2.imshow("inbuilt",matched_inbuilt)
		cv2.waitKey(0)

		diff_img=np.abs(matched_inbuilt.astype(np.int16)-matched_user_def)
		cv2.imshow("difference",diff_img)
		cv2.waitKey(0)