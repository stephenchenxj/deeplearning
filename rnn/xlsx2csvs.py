# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:08:07 2020

@author: stephen.chen
"""

import xlrd
import csv

def csv_from_excel():
    wb = xlrd.open_workbook('data\\stockData.xlsx')
    sheet_names = wb.sheet_names
    print(sheet_names())
    for sheet_name in sheet_names():
        print(sheet_name)
        
        sh = wb.sheet_by_name(sheet_name)
        your_csv_file = open('data\\'+ sheet_name +'.csv', 'w', newline="")
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_NONE)

        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))    
        your_csv_file.close()

# runs the csv_from_excel function:
csv_from_excel()