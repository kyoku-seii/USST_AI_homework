def judge_prime_composite(i, prime, composite):
    flag = True
    for j in range(2, i):
        if i % j == 0:
            flag = False
            break
    if flag and i != 1:
        prime.append(i)
    else:
        composite.append(i)


def judge_odd_even(number, odd, even):
    if number % 2 == 0:
        even.append(number)
    else:
        odd.append(number)


def find_prime_number(start, end):
    prime = []  # 素数
    composite = []  # 和数
    odd = []  # 奇数
    even = []  # 偶数

    for i in range(start, end + 1):
        judge_odd_even(i, odd, even)
        judge_prime_composite(i, prime, composite)
    return prime, composite, odd, even


def main():
    start = int(input("输入第一个数字"))
    end = int(input("输入第二个数字"))
    if end < start:
        print('第二个数一定要大于第一个数哟')
        return
    answer = find_prime_number(start, end)
    print("您输入的区间中质数的个数为" + str(len(answer[0])))
    print("他们是:"+str(answer[0]))
    print("您输入的区间中合数的个数为" + str(len(answer[1])))
    print("他们是:"+str(answer[1]))
    print("您输入的区间中奇数的个数为" + str(len(answer[2])))
    print("他们是:" + str(answer[2]))
    print("您输入的区间中偶数的个数为" + str(len(answer[3])))
    print("他们是:" + str(answer[3]))


if __name__ == "__main__":
    main()
