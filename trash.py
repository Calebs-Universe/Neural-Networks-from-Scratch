import asyncio
import time


async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)
    return 'done'


async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(
            say_after(1, 'Hello')
        )
        t2 = tg.create_task(
            say_after(2, 'Guys')
        )

        print(f'Started at {time.strftime('%X')}')

    print(f'Stopped at {time.strftime('%X')}')
    print(f'T1: {t1.result()}, T2: {t2.result()}')


asyncio.run(main())
