-- Write an sql query to find books that have sold fewer than 10 copies in the last year, excluding books that have been available for less than 1 month.

-- Product table
create table product
(
product_id number primary key,
name_ varchar2(128) not null,
rrp number not null,
available_from date not null
);
-- Order table
create table orders
(
order_id number primary key,
product_id number not null,
quantity number not null,
order_price number not null,
dispatch_date date not null,
foreign key (product_id) references product(product_id)
);

-- sql query
SELECT
product.name_ as name_
FROM
orders
INNER JOIN product ON product.product_id = orders.product_id
WHERE quantity < 10 AND dispatch_date >= date('now', '-1 years') AND product.available_from <= date('now', '-1 months');


-- Example:
-- Populate Product table
INSERT INTO product(product_id, name_, rrp, available_from)
VALUES (101, 'Bayesian Methods for Nonlinear Classification and Regression', 94.95, date('now','-6 days'));
INSERT INTO product(product_id, name_, rrp, available_from)
VALUES (102, '(next year) in Review (preorder)', 21.95, date('now','+1 years'));
INSERT INTO product(product_id, name_, rrp, available_from)
VALUES (103, 'Learn Python in Ten Minutes', 2.15, date('now','-3 months'));
INSERT INTO product(product_id, name_, rrp, available_from)
VALUES (104, 'sports almanac (1999-2049)', 3.38, date('now','-2 years'));
INSERT INTO product(product_id, name_, rrp, available_from)
VALUES (105, 'finance for dummies', 84.99, date('now','-1 years'));

-- Populate order table
INSERT INTO orders(order_id, product_id, quantity,order_price, dispatch_date)
VALUES (1000, 101, 1, 90.00, date('now','-2 months'));
INSERT INTO orders(order_id, product_id, quantity,order_price, dispatch_date)
VALUES (1001, 103, 1, 1.15, date('now','-40 days'));
INSERT INTO orders(order_id, product_id, quantity,order_price, dispatch_date)
VALUES (1002, 101, 10, 90.00, date('now','-11 months'));
INSERT INTO orders(order_id, product_id, quantity,order_price, dispatch_date)
VALUES (1003, 104, 11, 3.38, date('now','-6 months'));
INSERT INTO orders(order_id, product_id, quantity,order_price, dispatch_date)
VALUES (1004, 105, 11, 501.33, date('now','-2 years'));