@app.post("/parse_xml")
def parse_xml():
    xml_src = request.args.get('xml_src')
    # parse xml_src and return it as a string