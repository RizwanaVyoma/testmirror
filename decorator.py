# Demo file to check the decorator working
class my_decorator():
    def my_func(func):
        def wrapper(*args,**k):
            print("Before calling the function...")
            result = func(*args,**k)
            print("After calling the function...")
            return result
        return wrapper

@my_decorator.my_func
def my_function(name,count):
    for i in range(count):
        print("I am the function!"+name)

my_function("test",count=5)