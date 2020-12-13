# Description
A simple web application that takes in a ***photo*** 
of ultility bill and try to find various field, such as:
- Total payable amount
- Company name (not yet implemented)
- Pay date (not yet implemented)
- BSB and Ref no. (not yet implemented)

# Quick start
From read_bill_site folder, run pytest tests:

```pytest```

The tests will try to interpret 28 different
utility bills (electrical, water, gas, internet).
Not all of the tests will be successful.

Also from read_bill_site, run the Django server:

```python3 manage.py runserver```

Go to ```localhost:8000```. This will let 
the user upload a image and receive back the result.

```example.py``` also provides a playground.

# How it works?
The app depends heavily in Google's Tesseract project.
It will try to detect a field, for example "Total amount", through
finding some keywords (total, amount, direct debit, etc.).

After finding the location of a field, it will try to find a number
either on the right or below the field. If it fails to find any number
satisfies the number, it will find the closest number.

If it fails to find the any field (think of field as an anchor),
it will just use the largest number for "Total amount".

If all cases fails, the value will be None.

# Road map
- Increase detection accuracy
- Detect BPAY info  

