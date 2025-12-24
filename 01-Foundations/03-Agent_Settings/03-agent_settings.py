import sys
import time
from rich import print

# ==========================================
# SECTION 1: The Junior Dev's Dilemma (The Problem)
# ==========================================

def check_access_bad(user_role):
    # The "Or-Chain" Mess
    # Hard to read, hard to maintain, inefficient structure
    if user_role == 'admin' or user_role == 'editor' or user_role == 'moderator' or user_role == 'support_level_2' or user_role == 'content_manager':
        return True
    return False

# distinct visualization of the problem
current_role = 'support_level_2'
if check_access_bad(current_role):
    print(f"Access granted for {current_role} (Bad Method)")


# ==========================================
# SECTION 2: The Senior Dev's Secret (The Solution)
# ==========================================

def check_access_good(user_role):
    # Step 1: Data separation
    allowed_roles = ['admin', 'editor', 'moderator', 'support_level_2', 'content_manager']
    
    # Step 2: The Pythonic "any()" function
    # Readable, maintainable, and efficient
    if any(user_role == role for role in allowed_roles):
        return True
    return False

if check_access_good(current_role):
    print(f"Access granted for {current_role} (Good Method)")


# ==========================================
# SECTION 3 & 4: Performance & Lazy Evaluation
# ==========================================
# 

# Demonstration of Lazy Evaluation (The "Conveyor Belt")
# We will use a helper function to PROVE it stops early
def is_match(role, target):
    print(f"Checking: {role}...") # Side effect to visualize checking
    return role == target

allowed_roles = ['admin', 'editor', 'moderator', 'support_level_2', 'content_manager']
target_role = 'editor' # This is the 2nd item

print("\n--- Starting Lazy Evaluation Check ---")
# This generator expression yields values one by one
# Notice it stops printing "Checking..." once it hits 'editor'
if any(is_match(role, target_role) for role in allowed_roles):
    print("Access Granted!")

# Comparison: List Comprehension (Bad for memory/speed in this context)
# print("\n--- Starting List Comprehension (Greedy) Check ---")
# This would print "Checking..." for EVERY role before deciding
# if any([is_match(role, target_role) for role in allowed_roles]):
#     print("Access Granted!")


# ==========================================
# SECTION 5: When 'or' is Still Your Friend
# ==========================================
# 

# The Idiomatic Default Value Pattern
provided_name_empty = ""
provided_name_filled = "Alice"

# Logic: Returns the first "Truthy" value
user_name_1 = provided_name_empty or "Guest" 
user_name_2 = provided_name_filled or "Guest"

print(f"\nUser 1: {user_name_1}") # Output: Guest
print(f"User 2: {user_name_2}") # Output: Alice


# ==========================================
# SECTION 6: Common Pitfalls
# ==========================================

print("\n--- Pitfalls ---")

# Pitfall 1: Empty Iterable
# Returns False (consistent logic)
print(f"Any of empty list: {any([])}") 

# Pitfall 2: Dictionaries
# iterates over KEYS by default
my_dict = {'admin': False, 'editor': True}

# Checks if any KEY is truthy (strings are usually truthy)
print(f"Any (dict keys): {any(my_dict)}") 

# Checks if any VALUE is True
print(f"Any (dict values): {any(my_dict.values())}")