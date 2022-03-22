#!/usr/bin/env python3
"""
Created on Fri Sep 11 09:45:33 2020

@author: Chengyan Fan
"""
from xml.dom.minidom import parse
from data_downloader import downloader

####################################################################################################
#  在此修改输入输出文件路径
#########################

# 文件输出目录，需确保此文件夹存在
folder_out = r'E:\data\dwq\19y'
# 第一步下载的包含url的 products.meta4 文件
url_file = r'E:\data\dwq\19y\products.meta4'
####################################################################################################


data = parse(url_file).documentElement
urls = [i.childNodes[0].nodeValue for i in data.getElementsByTagName('url')]

downloader.download_datas(urls, folder_out)
# from data_downloader import downloader
#
# netrc = downloader.Netrc()
# netrc.add('scihub.copernicus.eu','jen3en','7482927m',overwrite=True)