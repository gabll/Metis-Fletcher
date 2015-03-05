### Cloned and adapted from https://github.com/roshansingh/python-tagcloud

from PIL import Image, ImageDraw, ImageFont
import math, os
import random
import uuid

class TagCloud(object):

	FONT = '/Library/Fonts/AppleGothic.ttf'
	#FONT_COLOR = ['#aab5f0', '#99ccee', '#a0ddff', '#90c5f0', '#90a0dd', '#90c5f0', '#39d', '#C8E1c2', '#88BCE2', '#0cf']
	#FONT_COLOR = ['#F2B701', '#E57D04', '#DC0030', '#B10058', '#7C378A', '#3465AA', '#09A275', '#85BC5F', '#39d', '#aab5f0']
	FONT_COLOR = ['#C3331C', '#CB1708', '#3B2821', '#AA7C13', '#8C8D3D', '#E6AC02', '#F5F24B', '#1A6802', '#4CAA52', '#aab5f0']
	FONT_SIZE = [15, 18, 20, 22, 24, 27, 30, 35, 40, 45]
				
	
	def __init__(self, width=400, height=400):
		self.width = width
		self.height = height
		self.words_to_draw = None
		self.image = Image.new('RGBA', [width, height], "#fff")
		self.imageDraw = ImageDraw.Draw(self.image)
		
	def formatWordList(self, words, values):
		if not isinstance(words, list):
			raise ValueError('words should be a list')
		
		if not isinstance(values, list):
			raise ValueError('values should be a list')
		
		formattedWords = []
		count = 0
		for word in words:
			formattedWords.append({"text": word, "weight": values[count]})
			count += 1
		
		return formattedWords
		
	def draw(self, words, imageFilePath=None):
		self.words = words
		if imageFilePath is None:
			imageFilePath = str(uuid.uuid4()) + '.jpg'
		self.imageFilePath = imageFilePath
		
		index = 0
		length = len(self.words)
		for word in self.words:
			if index == length - 1:
				weight = 0
			else:
				weight = self._rescaleWeight(word['weight'], self.words[0]['weight'], self.words[-1]['weight'])
			self._findCoordinates(index, word['text'], int(weight))
			index += 1
	
		return self._save()
	
	def _rescaleWeight(self, n, maxinum, minimum):
		scaleMin = 1
		scaleMax = 10
		
		# if max and min is same return max weight - 1
		if maxinum == minimum:
			return 9
		
		weight = round((1.0 * (scaleMax - scaleMin) * (n - minimum)) / (maxinum-minimum))
		
		return weight
	
	def _findCoordinates(self, index, text, weight):
		angleStep = 0.57
		radiusStep = 8
		radius = 25
		angle = random.uniform(0.2, 6.28)
		
		fontsize = self.FONT_SIZE[weight]
		width, height = self.imageDraw.textsize(text, font=ImageFont.truetype(self.FONT, fontsize))
		
		x = self.width/2 - width/2.0
		y = self.height/2 - height/2.0
		
		count = 1
		while self._checkOverlap(x, y, height, width):
			if count % 8 == 0:
				radius += radiusStep
			count += 1
			
			if index % 2 == 0:
				angle += angleStep
			else:
				angle += -angleStep
			
			x = self.width/2 - (width / 2.0) + (radius*math.cos(angle))
			y = self.height/2 + radius*math.sin(angle) - (height / 2.0)
		
		self.words_to_draw.append({'text': text, 'fontsize': fontsize, 'x': x, 'y': y, 'w': width, 'h': height, 'color': self.FONT_COLOR[weight]})
		
	def _checkOverlap(self, x, y, h, w):
		if not self.words_to_draw:
			self.words_to_draw = []
			return False
		
		for word in self.words_to_draw:
			#if overlaps
			if not ((x+w < word['x']) or (word['x'] + word['w'] < x) or (y + h < word['y']) or (word['y'] + word['h'] < y)):
				return True
		
		return False
	
	def _save(self):
		for word in self.words_to_draw:
			if self._liesInside(word):
				self.imageDraw.text((word['x'], word['y']), word['text'], font=ImageFont.truetype(self.FONT, word['fontsize']), fill=word['color'])
			
		self.image.save(self.imageFilePath, "JPEG", quality=100)
		return self.imageFilePath

	def _liesInside(self, word):
		'''
		if lies within the final image size
		'''
		
		if word['x'] >= 0 and word['x'] + word['w'] <= self.width and word['y'] >= 0 and word['y'] + word['h'] <= self.height:
			return True
		
		return False 
		
		
if __name__ == '__main__':
	t = TagCloud()
	words = [{"text": "coffee", "weight": 20296.0}, {"text": "love", "weight": 15320.0}, {"text": "day", "weight": 6860.0}, {"text": "like", "weight": 5521.0}, {"text": "follow", "weight": 5393.0}, {"text": "morning", "weight": 5125.0}, {"text": "happy", "weight": 5099.0}, {"text": "girl", "weight": 5049.0}, {"text": "cute", "weight": 4336.0}, {"text": "good", "weight": 4328.0}, {"text": "tumblr", "weight": 4169.0}, {"text": "today", "weight": 4142.0}, {"text": "followme", "weight": 3923.0}, {"text": "chocolate", "weight": 3922.0}, {"text": "instagood", "weight": 3818.0}, {"text": "yummy", "weight": 3786.0}, {"text": "new", "weight": 3700.0}, {"text": "lol", "weight": 3536.0}, {"text": "yum", "weight": 3282.0}, {"text": "drink", "weight": 3246.0}, {"text": "latte", "weight": 3219.0}, {"text": "time", "weight": 3137.0}, {"text": "caramel", "weight": 3125.0}, {"text": "friends", "weight": 3084.0}, {"text": "tagsforlikes", "weight": 3005.0}, {"text": "beautiful", "weight": 2892.0}, {"text": "food", "weight": 2801.0}, {"text": "life", "weight": 2780.0}, {"text": "delicious", "weight": 2767.0}, {"text": "f4f", "weight": 2670.0}, {"text": "follow4follow", "weight": 2667.0}, {"text": "white", "weight": 2614.0}, {"text": "tea", "weight": 2577.0}, {"text": "selfie", "weight": 2559.0}, {"text": "best", "weight": 2522.0}, {"text": "swag", "weight": 2494.0}, {"text": "got", "weight": 2491.0}, {"text": "work", "weight": 2479.0}, {"text": "fashion", "weight": 2458.0}, {"text": "likeforlike", "weight": 2440.0}, {"text": "amazing", "weight": 2434.0}, {"text": "followforfollow", "weight": 2366.0}, {"text": "get", "weight": 2365.0}, {"text": "fun", "weight": 2273.0}, {"text": "like4like", "weight": 2236.0}, {"text": "frappuccino", "weight": 2167.0}, {"text": "picoftheday", "weight": 2083.0}, {"text": "breakfast", "weight": 2062.0}, {"text": "smile", "weight": 2060.0}, {"text": "photooftheday", "weight": 2016.0}, {"text": "summer", "weight": 1982.0}, {"text": "hot", "weight": 1928.0}, {"text": "mocha", "weight": 1906.0}, {"text": "instadaily", "weight": 1896.0}, {"text": "pink", "weight": 1883.0}, {"text": "perfect", "weight": 1804.0}, {"text": "shopping", "weight": 1801.0}]
	
	print t.draw(words)