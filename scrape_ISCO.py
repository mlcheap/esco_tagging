# scrapy runspider scrape_ISCO.py -o ISCO_codes.jl

import scrapy
import requests
import psycopg2 as pg
import pandas.io.sql as psql

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    def start_requests(self):
        connection = pg.connect(user="postgres",
                        # password='daynasor',
                        host="127.0.0.1",
                        port="5432",
                        database="mydb2")
        urls = psql.read_sql('SELECT external_id FROM occupations', connection).values
        urls = [url[0] for url in urls if (isinstance(url[0],str) and (url[0][:4]=='http'))]
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        ISCO_code = response.css("#dataContainer > article > div > p:nth-child(2)::text").get()
        self.log(f'ISCO code {ISCO_code}')
        url = requests.utils.unquote(response.url)
        url = url.split('&')[-3].split('=')[-1]
        yield {url:ISCO_code}
