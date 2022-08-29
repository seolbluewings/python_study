
### Python Class ###
### - StarCraft Terran 유닛을 만들고자 하는데 Class 선언 없이 진행한다면?

'''마린 : 공격 유닛, 군인'''
name = 'marine' # Unit 이름
hp = 40 # Unit HP
damage = 6 # Unit 공격력
print("{} 유닛이 생성되었습니다.".format(name))
print("체력 {0}, 공격력 {1}\n".format(hp,damage))


'''시즈 탱크 : 공격 유닛, 탱크, 일반/시즈 모드'''
tank_name = 'siege_tank'
tank_hp = 150
tank_damage = 30
print("{} 유닛이 생성되었습니다.".format(tank_name))
print("체력 {0}, 공격력 {1}\n".format(tank_hp,tank_damage))


'''공격 행위에 대한 함수 정의'''
def attack(name, location, damage) :
    print("{0} : {1} 방향으로 적군을 공격 합니다. [공격력{2}]".format(name,location,damage))
    
attack(name,"5시",damage)
attack(tank_name,"5시",tank_damage)


### 이런 방식으로 정의한다면 유닛이 하나 추가되었을 때, 계속해서 변수를 만드는 것이 필요한데 굉장히 비효율적

### 만약 위의 방식을 Class로 바꾼다면? ###

class Unit :
    def __init__(self, name, hp, damage) :
        self.name = name
        self.hp = hp
        self.damage = damage
        print("{0} 유닛이 생성되었습니다.".format(self.name))
        print("체력 {0}, 공격력 {1}".format(self.hp, self.damage))

marine1 = Unit("marine",40,6)
marine2 = Unit("marine",40,6)
seize_tank1 = Unit("Seize_Tank",150,30)
seize_tank2 = Unit("Seize_Tank",150,30)

print("유닛 이름: {0}, 공격력: {1}".format(marine1.name, marine1.damage))


### Method 정의 ###

class AttackUnit:    
    count = 0
        
    def __init__(self, name, hp, damage) :
        self.name = name
        self.hp = hp
        self.damage = damage
        AttackUnit.count += 1
    
    def unit_count(cls) :
        print("유닛이 {}개 생성되었습니다".format(cls.count))
    
    def attack(self, location) :
        print("{0} : {1} 방향으로 적군을 공격 합니다. [공격력 {2}]".format(self.name, location, self.damage))
    
    def damaged(self, damage) :
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력은 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0 :
            print("{0} : 유닛이 파괴되었습니다.".format(self.name))

### 다른 유닛을 정의 ###
firebat1 = AttackUnit("firebat",50,16)
AttackUnit.unit_count()
firebat1.attack("5시")
firebat1.damaged(25)


### 상속, 다중상속 ###
### Unit, AttackUnit 에서 self.name, self.hp는 동등
### Unit 클래스를 상속받아 AttackUnit이란 클래스를 만드는 것으로 코드 변경

class Unit :
    def __init__(self, name, hp) :
        self.name = name
        self.hp = hp

class AttackUnit(Unit) :
        
    def __init__(self, name, hp, damage) :
        Unit.__init__(self,name,hp)
        self.damage = damage
        
    def attack(self,location) :
        print("{0} : {1} 방향으로 적군을 공격합니다. [공격력 {2}]"             .format(self.name, location, self.damage))
    
    def damaged(self, damage) :
        print("{0} : {1} 데미지를 입었습니다.".format(self.name, damage))
        self.hp -= damage
        print("{0} : 현재 체력은 {1} 입니다.".format(self.name, self.hp))
        if self.hp <= 0 :
            print("{0} : 유닛이 파괴되었습니다.".format(self.name))
            
            
### 비행 기능에 대한 Class 선언
class Flyable :
    def __init__(self, flying_speed) :
        self.flying_speed = flying_speed
    
    def fly(self, name, location) :
         print("{0} : {1} 방향으로 날아갑니다합니다. [속도 {2}]"             .format(self.name, location, self.flying_speed))

### 공중 공격 유닛 Class 선언 (Wraith 등...)
### 다중상속 시, 콤마(,)로 구분
class Flyable_AttackUnit(AttackUnit, Flyable) :
    def __init__(self, name, hp, damage, flying_speed) :
        AttackUnit.__init__(self, name, hp, damage)
        Flyable.__init__(self,flying_speed)


# 공중 공격 유닛인 Valkyrie 생성
valkyrie = Flyable_AttackUnit("Valkyrie",200,6,5)
valkyrie.fly(valkyrie.name,"4시")

vulture = AttackUnit("vulture",80,10,20)
battlecruiser = Flyable_AttackUnit("battlecruiser",500,25,3)