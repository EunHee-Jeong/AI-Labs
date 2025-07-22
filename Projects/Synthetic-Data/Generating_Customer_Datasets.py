from faker import Faker
import faker.providers.credit_card
import pandas as pd

import random
from random import randint

# 1. Import helper-function
def random_n_digits(n):
	range_start = 10**(n-1)
	range_end = (10**n)-1
	return randint(range_start, range_end)

def unique_rand(rands, n):
	new_int = random_n_digits(n)
	if new_int not in rands:
		rands.append(new_int)
	else:
		unique_rand(rands, n)
	return new_int, rands

def generate_cost():
	cost = ''
	digits = randint(1, 4)
	cost += str(random_n_digits(digits))
	cost += '.'+str(random_n_digits(2))
	return cost

# 2. Creating Customers
fake = Faker(['en_US', 'en_UK', 'it_IT', 'de_DE', 'fr_FR'], use_weighting=True)

customers = {}
cust_ids = []

for i in range(0, 10000):
	customers[i] = {}
	customers[i]['cust_id'], cust_ids = unique_rand(cust_ids, 8)
	customers[i]['name'] = fake.name()
	customers[i]['address'] = fake.address().replace('\n', ', ')
	customers[i]['phone_number'] = fake.phone_number()
	customers[i]['dob'] = fake.date()
	customers[i]['note'] = fake.text().replace('\n', ' ')

customer_df = pd.DataFrame(customers).T
print(customer_df)

customer_df.to_csv('customer_data.csv', index=False)

credit_cards = {}
cc_ids = []

for i in range(0, 10000):
	credit_cards[i] = {}
	credit_cards[i]['cc_id'], cc_ids = unique_rand(cc_ids, 5)
	credit_cards[i]['type'] = fake.credit_card_provider()
	credit_cards[i]['number'] = fake.credit_card_number()
	credit_cards[i]['ccv'] = fake.credit_card_security_code()
	credit_cards[i]['expire'] = fake.credit_card_expire()

credit_cards_df = pd.DataFrame(credit_cards).T
print(credit_cards_df)

credit_cards_df.to_csv('credit_card_data.csv', index=False)

orders = {}
order_ids = []

for i in range(0, 1000):
	orders[i] = {}
	orders[i]['order_id'], order_ids = unique_rand(order_ids, 10)
	orders[i]['cust_id'] = random.choice(cust_ids)
	orders[i]['cc_id'] = random.choice(cc_ids)
	orders[i]['cost'] = generate_cost()

orders_df = pd.DataFrame(orders).T
print(orders_df)

orders_df.to_csv('orders.csv', index=False)