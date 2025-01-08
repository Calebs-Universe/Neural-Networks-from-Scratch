def say_hello(*args, **kwargs):
    # print(txt)
    # fn()
    print(f'{args} {kwargs}')
    print('COol')

@say_hello('Hello')
def cool():
    print("how ma boy")