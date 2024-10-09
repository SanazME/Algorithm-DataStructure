## Decorators
- https://realpython.com/primer-on-python-decorators/
- https://stackoverflow.com/questions/308999/what-does-functools-wraps-do

## Interview:
- fix this poorly written code.
```py
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/get_user_orders', methods=['GET'])
def get_user_orders():
    user_id = request.args.get('user_id')
    if user_id:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM orders WHERE user_id=?", (user_id,))
        orders = cur.fetchall()
        conn.close()
        return jsonify(orders)
    else:
        return jsonify({'error': 'User ID is required'}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

**Answer**:
- What are Functional requirement and also Non-functional requirement? (scalability, performance etc)

- Areas for improvement:
1. Error handling
2. GET Resource name
   - scalability and maintainence: we need to version the API `/api/v1/...` to allow for future updates without breaking existing clients.
   - API response format and consistency: make sure we're consistent in using json for all responses.
   - returned correct HTTP codes 404 (not found), 500 etc
4. Logging
5. Database connection creation and management
   - scalability and maintainence: we use a decorator for database connection, making it easier to switch to a connection pool or ORM in the future if needed.
   - we separate concerns by creating functions for database connection and using decorators.
   - Performance: we just nee to query needed fields and not all, also limit how much we query LIMIT and we need to make **Batch** query.


```py
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sqlite3
import logging
from functools import wraps

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

# Decorator for database connection management
def with_db_connection(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        conn = get_db_connection()
        try:
            return f(conn, *args, **kwargs)
        finally:
            conn.close()
    return decorated_function

@app.route('/api/v1/users/<int:user_id>/orders', methods=['GET'])
@limiter.limit("10 per minute")
@with_db_connection
def get_user_orders(conn, user_id):
    try:
        # Input validation
        if not isinstance(user_id, int) or user_id <= 0:
            return jsonify({'error': 'Invalid user ID'}), 400

        # Query execution
        cur = conn.cursor()
        cur.execute("SELECT id, product_name, quantity, price, order_date FROM orders WHERE user_id = ?", (user_id,))
        orders = [dict(row) for row in cur.fetchall()]

        if not orders:
            return jsonify({'message': 'No orders found for this user'}), 404

        # Successful response
        return jsonify({
            'user_id': user_id,
            'order_count': len(orders),
            'orders': orders
        }), 200

    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False)
```
