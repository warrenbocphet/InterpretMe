from django.shortcuts import render

from process.process import Extractor


def index(request):
    if request.method == "POST":
        file = request.FILES["image"]

        if file.content_type[:5] != "image":
            return render(request, r"read_bill/index.html", context={"total_payable": "unsupported type"})

        b = file.read()

        ext = Extractor(b)
        ext.extract()

        if ext.fields["total_payable"] is None:
            return render(request, r"read_bill/index.html", context={"total_payable": "undetectable"})

        return render(request, r"read_bill/index.html", context=ext.fields)

    else:
        return render(request, r"read_bill/index.html")
