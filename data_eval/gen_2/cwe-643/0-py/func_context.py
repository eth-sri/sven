def run_xpath_query(request):
    tagid = request.GET['tagid']
    tree = etree.parse(StringIO('<foo><bar></bar></foo>'))
    # construct an XPath object for tags whose id equals tagid
    r = tree.xpath("/tag[@id=$tagid]", tagid=value)