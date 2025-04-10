from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'yoghurt'})
crawler.crawl(keyword='yoghurt cup', max_num=50)