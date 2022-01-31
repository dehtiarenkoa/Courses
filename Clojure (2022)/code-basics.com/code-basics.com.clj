;tasks from https://ru.code-basics.com/languages/clojure/
;from there also commented out tests and alternative variants of the right answers


;--1)
(println "Hello  world")


;--2)
;Выведите в стандартный поток вывода (с помощью функции println) следующее выражение: 1 + 10 - 2 * 7
(println (+ 1 (- 10 (* 2 7))))


;--3)
;Выведите в стандартный поток вывода (с помощью функции println) результат вычисления следующего выражения: - 23 * (- 3) + 15
(println (+ (*  (- 23) (- 3)) 15))


;--4)
;Выведите в стандартный поток вывода (с помощью функции println) следующее выражение: 100 - 34 - 22 - (5 + 3 - 10)
(println (- 100 34 22 (- (+ 5 3) 10)))


;--5)
;Выведите в стандартный поток вывода (с помощью функции println) результат парсинга строки 256 
;(с помощью метода Integer/parseInt) .
(println (Integer/parseInt "256"))
;test:
;(ns jvm-errors-test
;   (:require [test-helper :refer [assert-output]]))
;(assert-output "256")


;--6)
;Создайте объявление, обозначающее "количество участников" (имя соорудите сами), 
;присвойте ему значение 10 и распечатайте на экран.
(def name 10)
(println name)


;--7)
;Создайте объявление, обозначающее "количество пар" (имя соорудите сами), 
;присвойте ему значение 5, затем с помощью формы defonce переопределите 
;количество пар на любое другое значение и распечатайте на экран.
(def nam 5)
(defonce nam "j")
(println nam)


;--8)
;Создайте функцию с именем square, которая вычисляет квадрат переданного числа
;(square 3) ; 9
(defn square [x] (* x x))
(square 3)


;--9)
;Определите (без создания переменной) и вызовите функцию, 
;которая находит среднее арифметическое между двумя числами. 
;В качестве чисел подставьте 2 и 4.
;Запишите результат в переменную.
;Выведите переменную на экран.
(def n ((fn [x y] (/ (+ x y) 2)) 2 4))
(println n)


;--10)
;Создайте функцию с именем sum-of-squares (используя короткий синтаксис), 
;которая находит сумму квадратов двух чисел.
;(sum-of-squares 2 3) ; 13
(defn sum-of-squares [a b] (+ (* a a) (* b b)))
(sum-of-squares 2 3)


;--11)
;Определите константу phone со значением "iphone"
(defonce phone "iphone")


;--12)
;Реализуйте функцию prod-sum, которая сначала умножает переданное число на себя, 
;а затем суммируется между собой и полученным результатом умножения. 
;Воспользуйтесь локальными объявлениями для хранения промежуточных результатов вычисления.
;(prod-sum 2) ; 6
;(prod-sum 3) ; 12
;(prod-sum 4) ; 20
(defn prod-sum [y] (letfn ([prod [x] (* x x)]) (+ y (prod y))))
(prod-sum 2)


;--13)
;Реализуйте функцию leap-year?, которая проверяет, является ли год високосным. 
;Любой год, который делится на 4 без остатка, является високосным годом. 
;Тем не менее, есть еще небольшая особенность, которая должна быть учтена например, 
;григорианский календарь предусматривает, что год, который делится без остатка на 100 
;является високосным годом только в том случае, если он также без остатка делится на 400.
;(leap-year? 2012) ; true
;(leap-year? 1913) ; false
;(leap-year? 1804) ; true
;(leap-year? 2100) ; false
(defn leap-year? [n]
  (letfn
   [(rem4 [n] (== 0 (rem n 4)))
    (rem100 [n] (== 0 (rem n 100)))
    (rem400 [n] (== 0 (rem n 400)))]
    (and
     (rem4 n)
     (not
      (and
       (rem100 n)
       (not (rem400 n)))))))
;(clojure.repl/doc rem)
(leap-year? 2012) ; true
(leap-year? 1913) ; false
(leap-year? 1804) ; true
(leap-year? 2100) ; false
;answer:
;; (ns index)
;; ;BEGIN
;; (defn leap-year? [year]
;;   (letfn [(divisible? [a b] (zero? (mod a b)))]
;;     (and (divisible? year 4) (or (not (divisible? year 100)) (divisible? year 400)))))
;; ;END


;--14)
;Реализуйте функцию sentence-type, которая возвращает строку "cry", если переданый текст 
;написан заглавными буквами, и возвращает строку "common" в остальных случаях.
;(sentence-type "HOW ARE YOU?") ; "cry"
;(sentence-type "Hello, world!") ; "common"
;Для перевода строки в верхний регистр используйте функцию upper-case.
;(ns index  (:require [clojure.string :refer [upper-case]]))
(ns code-basics.com (:require [clojure.string :refer [upper-case]]))
(defn sentence-type [s] (if (= s (upper-case s)) "cry" "common"))
    ;; (letfn [(c [x] (every? #(Character/isAlphabetic %) x))
    ;;         (ch [x] (every? (#(and (Character/isAlphabetic %) (Character/isUpperCase  %)) x)))] 
    ;;   (if (ch s) "cry" "common")))
;(every? #(Character/isUpperCase %) "Hello, World!")
;(every? Character/isUpperCase "jjjx")
(sentence-type "HOW ARE YOU?") ; "cry"
(sentence-type "Hello, world!") ; "common"
(clojure.repl/doc isUpper)
;test
;; (ns if-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [sentence-type]]))
;; (assert-solution
;;  [["HOW?"] ["HoW?"] ["clojure"] ["CLOJURE"]]
;;  ["cry" "common" "common" "cry"]
;;  sentence-type)


;--15)
;Реализуйте функцию say-boom, которая возвращает строку Boom!, если её вызвали с параметром "go" .
;(say-boom "hey")
;(say-boom "go") ; "Boom!"
(defn say-boom [s] (when (= s "go")   "Boom!"))
(say-boom "hey")
(say-boom "go") ; "Boom!"
;test
;; (ns guards-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [say-boom]]))
;; (assert-solution [["hey"] ["go"]] [nil "Boom!"] say-boom)


;--16)
;Реализуйте функцию humanize-permission, которая принимает на вход символьное обозначение прав доступа в Unix системах, 
;и возвращает текстовое описание.
;(humanize-permission "x") ; execute
;(humanize-permission "r") ; read
;(humanize-permission "w") ; write
(defn humanize-permission [t] (case t "x" "execute", "r" "read", "w" "write"))
(humanize-permission "x") ; execute
(humanize-permission "r") ; read
(humanize-permission "w") ; write
;test
;; (ns case-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [humanize-permission]]))
;; (assert-solution [["x"] ["r"] ["w"]] ["execute" "read" "write"] humanize-permission)


;--17)
;Реализуйте функцию programmer-level, которая принимает на вход количество баллов, и возвращает уровень 
;разработчика на основе этого числа. Если баллов от 0 до 10, то это junior, от 10 до 20 – middle, от 20 и выше – senior.
;(programmer-level 10) ; middle
;(programmer-level 0) ; junior
;(programmer-level 40) ; senior
(defn programmer-level [x] (cond (and (< x 10) (>= x 0)) "junior"
                                 (and (< x 20) (>= x 10)) "middle"
                                 (and (<= x 40) (>= x 20)) "senior")) ; middle
(programmer-level 10) ; middle
(programmer-level 0) ; junior
(programmer-level 40) ; senior
;answer
;; (defn programmer-level [points-count]
;;   (cond
;;     (< points-count 10) "junior"
;;     (and (>= points-count 10) (< points-count 20)) "middle"
;;     :else "senior"))
;test
;; (ns cond-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [programmer-level]]))
;; (assert-solution
;;  [[3] [18] [40]]
;;  ["junior" "middle" "senior"]
;;  programmer-level)


;--18)
;Реализуйте функцию do-today, которая принимает порядковый номер дня недели (целое число) в качестве аргумента и вычисляется в
;- строку "work" для дней с понедельника (1) по пятницу (5)
;- строку "rest" для субботы (6) и воскресенья (7)
;- строку "???" для всех остальных значений, в том числе и для нечисловых!
;Попробуйте использовать в решении различные комбинации if, cond и case.
;Советы: Используйте функцию-предикат int? чтобы проверить, что аргумент — целое число.
(defn do-today [n] (cond
                     (int? n) (case n
                                (1 2 3 4 5) "work"
                                (6 7) "rest"
                                "???")
                     :else "???"))
(do-today 44)
;test
;; (ns expressions-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [do-today]]))
;; (assert-solution
;;  [[1] [2] [3] [4] [5] [6] [7] [0] [-1] [10] [false] ["oops"]]
;;  ["work" "work" "work" "work" "work" "rest" "rest" "???" "???" "???" "???" "???"]
;;  do-today)


;--19)
;Реализуйте функцию triple, которая должна принимать один аргумент любого типа и возвращать список, 
;в котором содержится три копии аргумента:
;(triple "a") ; '("a" "a" "a")
;(triple 0)   ; '(0 0 0)
(defn triple [t] (list t t t))
(triple "a") ; '("a" "a" "a")
(triple 0)   ; '(0 0 0)
;test
;; (ns intro-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [triple]]))
;; (assert-solution
;;  [["a"] [0] [true]]
;;  ['("a" "a" "a") '(0 0 0) '(true true true)]
;;  triple)


;--20)
;Реализуйте функцию maps, которая должна принимать два списка — список функций и 
;список списков аргументов — и возвращать список результатов применения функций к 
;наборам аргументов. Вот как использование maps должно выглядеть:
;(maps
;  (list
;   inc
;   string?)
;  (list
;   (list 10 20)
;   (list "a" 0)))
;; '((11 21) (true false))
;Здесь
;- '(11 21), это результат применения inc к (list 10 20)
;- '(true false), это результат применения string? к (list "a" 0)
;;
;; (def d1 [:a :b :c])
;; (#(map list % (range)) d1)
;; ;;=> ((:a 0) (:b 1) (:c 2))
;;
;(defn maps [x y] (list ((first x) (first y)) (maps (rest x) (first y)) )) ;err
;(defn maps [x y] (list  (map (first x) (first y)) (maps (rest x) (rest y)))) ;err
;(defn maps [x y] (list  (map x y))) ;err
;(defn maps [x y]  (defn fuun [a b] (a b)) (map fuun x y)) ;err
;(defn maps [x y]  (#(map list % y) x) )
;((#function[clojure.core/inc] (10 20)) (#function[clojure.core/string?--5427] ("a" 0)))
;(defn maps [x y]  (map #(%1 %2) x y)) ;err
; Error printing return value (ClassCastException) at clojure.lang.Numbers/inc (Numbers.java:137).
; clojure.lang.PersistentList cannot be cast to java.lang.Number
;(defn maps [x y] (#(map %1 %2) (vector x) (vector y)))
; Error printing return value (IllegalArgumentException) at clojure.lang.APersistentVector/null (APersistentVector.java:294).
; Key must be integer
;(defn maps [x y] (list map #(%1 %2) x y))
;(#function[clojure.core/map] #function[code-basics.com/maps/fn--7367] (#function[clojure.core/inc] #function[clojure.core/string?--5427]) ((10 20) ("a" 0)))
(defn maps [x y]  (map #(map %1 %2) x  y));(map #(%1 %2) x y))
(maps
 (list
  inc
  string?)
 (list
  (list 10 20)
  (list "a" 0)))
'((11 21) (true false))
;answer
;(defn maps [fs as] (map map fs as))
;; ;test
;; (ns loops-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [maps]]))
;; (assert-solution
;;  [[(list) (list)]
;;   [(list inc) (list (list 0))]
;;   [(list inc string?) (list (list 0 100) (list "foo" 42))]]
;;  [(list)
;;   (list (list 1))
;;   (list (list 1 101) (list true false))]
;;  maps)


;--21)
;Реализуйте функцию increment-numbers, которая берёт из списка-аргумента значения, являющиеся числами (number?) 
;и возвращает список этих чисел, увеличив предварительно каждое число на единицу (inc). Пример:
;(increment-numbers (list 10 "foo" false (list 2 3) 3/5)) ; '(11 8/5)
;Заметьте, Clojure умеет работать с дробями вроде 3/5 и 8/5!
(defn increment-numbers [x] (map inc (filter number? x)))
(increment-numbers (list 10 "foo" false (list 2 3) 3/5)) ; '(11 8/5)
;test
;; (ns filters-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [increment-numbers]]))
;; (assert-solution
;;  [['()] ['("a" "b" false)] ['(1 1/2 "foo")]]
;;  ['() '() '(2 3/2)]
;;  increment-numbers)


;--22)
;Реализуйте функцию max-delta, которая должна принимать два списка чисел и вычислять максимальную разницу 
;(абсолютное значение разницы) между соответствующими парами элементов.
;Пример использования:
;(max-delta
;  (list 10 -15 35)
;  (list 2 -12 42)) ; 8
;Вам пригодятся функции Math/abs и max:
;(Math/abs 42)    ; 42
;(Math/abs -13)   ; 13
;(max 1 5 3) ; 5
(defn max-delta [a b] (reduce max 0 (map #(Math/abs (- %1 %2)) a b)))
(max-delta
 (list 10 -15 35)
 (list 2 -12 42)) ; 8
;answer
;; (defn max-delta [xs ys]
;;   (reduce (fn [acc [x y]] (max acc (Math/abs (- x y))))
;;           0 (map list xs ys)))
;test
;; (ns reduce-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [max-delta]]))
;; (assert-solution
;;  [['() '()] ['(-5) '(-15)] ['(0) '(42)] ['(10 -15 35) '(2 -12 42)]]
;;  [0 10 42 8]
;;  max-delta)


;--23)
;Реализуйте функцию lookup, которая бы должна принимать аргумент-ключ и список пар "ключ-значение" и возвращать 
;либо пару "ключ-значение", где ключ равен первому аргументу, либо возвращать false, если подходящих пар в списке 
;не нашлось. Если подходящих пар найдётся несколько, вернуть нужно первую.
;Примеры:
;(def user-ages
;  (list (list "Tom" 31)
;        (list "Alice" 22)
;        (list "Bob" 42)))
;(lookup "Bob" user-ages) ; '("Bob" . 42)
;(lookup "Tom" user-ages) ; '("Tom" . 31)
;(lookup "Moe" user-ages) ; false

(defn lookup [k s]
  (defn fu [k s] (if (= nil (first (first s))) -1
                     (if (= k (first (first s))) (last (first s))
                         (fu k (rest s)))))
  (def v (fu k s))
  (println v)
  (if (= -1 v) false (list k v)))
(def user-ages
  (list (list "Tom" 31)
        (list "Alice" 22)
        (list "Bob" 42)))
(def r (list (list 42 0)
             (list 30 0)
             (list 42 100500)))
(lookup "Bob" user-ages) ; '("Bob" . 42)
(lookup "Tom" user-ages) ; '("Tom" . 31)
(lookup "Moe" user-ages) ; false
(lookup 42 r)
(lookup "foo"  (list (list "foo" "bar")))
;answer
;; (defn lookup [key pairs]
;;   (letfn [(same-key? [kv] (= key (first kv)))]
;;     (let [found-pairs (filter same-key? pairs)]
;;       (if (empty? found-pairs)
;;         false
;;         (first found-pairs)))))
;tests
;; (ns list-internals-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [lookup]]))
;; (assert-solution
;;  [["foo" nil]
;;   ["foo" (list (list "foo" "bar"))]
;;   ["foo" (list (list "bar" "foo"))]
;;   [42 (list (list 42 0)
;;             (list 30 0)
;;             (list 42 100500))]]
;;  [false '("foo" "bar") false '(42 0)]
;;  lookup)


;--24)
;Реализуйте функцию skip, которая должна принимать два аргумента — целое число n и список — и 
;возвращать новый список, содержащий все элементы из первого списка за исключением n первых элементов. 
;Если n окажется больше, чем количество элементов во входном списке, результатом должен быть пустой список.
;Примеры:
;(skip -5 (list 1 2 3)) ; '(1 2 3)
;(skip  0 (list 1 2 3)) ; '(1 2 3)
;(skip  1 (list 1 2 3)) ; '(2 3)
;(skip 10 (list 1 2 3)) ; '()
(defn skip [n s]
  (defn sk [n s acc]
    (def t (- n acc))
    (cond
      (= t 0) s
      (< t 0) (if (< n 0) s ())
      :else (recur n (rest s) (inc acc))))
  (sk n s 0))
(skip -5 (list 1 2 3)) ; '(1 2 3)
(skip  0 (list 1 2 3)) ; '(1 2 3)
(skip  1 (list 1 2 3)) ; '(2 3)
(skip 10 (list 1 2 3)) ; '()
;tests
;; (ns recur-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [skip]]))
;; (assert-solution
;;  [[-5 '(1 2 3)] [0 '(1 2 3)] [1 '(1 2 3)] [10 '(1 2 3)]]
;;  ['(1 2 3) '(1 2 3) '(2 3) '()]
;;  skip)
;answer
;; (defn skip [n l]
;;   (if (or (<= n 0) (empty? l)) l
;;       (skip (dec n) (rest l))))


;--25)
;Реализуйте функцию str-reverse, которая должна принимать вектор строк и перевернуть каждую:
;(str-reverse ["my" "str"]) ; ["ym" "rts"]
;(str-reverse [])           ; []
(ns code-basics.com  (:require [clojure.string :as s]))
(defn str-reverse [b] (mapv s/reverse  b))
(str-reverse ["my" "str"]) ; ["ym" "rts"]
(str-reverse [])           ; []
;tests
;; (ns intro-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [str-reverse]]))
;; (assert-solution
;;  [[["my" "str"]] [[]] [["hello" "world" "foo" "!"]]]
;;  [["ym" "rts"] [] ["olleh" "dlrow" "oof" "!"]]
;;  str-reverse)


;--26)
;Реализуйте функцию next-chars, которая создаёт новую строку на основе строки-аргумента таким образом, 
;что каждый символ новой строки является "следующим" (с точки зрения кода) по отношению к соответствующему 
;символу исходной строки.
;Примеры:
;(next-chars "")      ; ""
;(next-chars "abc")   ; "bcd"
;(next-chars "12345") ; "23456"
(defn next-chars [s] (apply str (map (comp char inc int) (seq s))))
(next-chars "")      ; ""
(next-chars "abc")  ; "bcd"
(next-chars "12345") ; "23456"
;answer
;; (ns index  (:require [clojure.string :as s]))
;; (defn next-chars [string]
;;   (->> string
;;        (seq)
;;        (map int)
;;        (map inc)
;;        (map char)
;;        (s/join #"")))
;test
;; (ns intro-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [next-chars]]))
;; (assert-solution
;;  [["abc"] [""] ["12345"]]
;;  ["bcd" "" "23456"]
;;  next-chars)


;--27)
;Реализуйте функцию number-presenter, которая представляет число в нескольких форматах.
;Примеры:
;(number-presenter 63); => "decimal 63  octal 77  hex 3f  upper-case hex 3F"
;(number-presenter 14); => "decimal 14  octal 16  hex e  upper-case hex E"
(defn number-presenter [n] (format "decimal %d  octal %o  hex %x  upper-case hex %X" n  n n n))
(number-presenter 63); => "decimal 63  octal 77  hex 3f  upper-case hex 3F"
(number-presenter 14); => "decimal 14  octal 16  hex e  upper-case hex E"
;tests
;; (ns intro-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [number-presenter]]))
;; (assert-solution
;;  [[63] [14] [2]]
;;  ["decimal 63  octal 77  hex 3f  upper-case hex 3F"
;;   "decimal 14  octal 16  hex e  upper-case hex E"
;;   "decimal 2  octal 2  hex 2  upper-case hex 2"]
;;  number-presenter)


;--28)
;Реализуйте функцию zip, которая группирует элементы переданных векторов в подвектора.
;Примеры:
;(zip [] []); => []
;(zip [1 2 3 4] [5 6 7 8]) ; => [[1 5] [2 6] [3 7] [4 8]]
;(zip [1 2] [3 4]); => [[1 3] [2 4]]
(defn zip [x y] (mapv vector x y))
(zip [] []); => []
(zip [1 2 3 4] [5 6 7 8]) ; => [[1 5] [2 6] [3 7] [4 8]]
(zip [1 2] [3 4]); => [[1 3] [2 4]]
;tests
;; (ns intro-vectors-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [zip]]))
;; (assert-solution
;;  [[[1 2 3 4] [5 6 7 8]] [[] []] [[1 2] [3 4]]]
;;  [[[1 5] [2 6] [3 7] [4 8]] [] [[1 3] [2 4]]]
;;  zip)


;--29)
;Реализуйте функцию sum, которая суммирует все элементы вектора.
;Примеры:
;(sum [])         ; => 0
;(sum [10 -20])   ; => -10
;(sum [1 2 3 4])  ; => 10
(defn sum [v] (reduce + (into '()  (reverse v))))
;  (map identity [1 2 3]);=>(1 2 3)
;  (apply list [1 2 3]);=>(1 2 3)  
(sum [])         ; => 0
(sum [10 -20])   ; => -10
(sum [1 2 3 4])  ; => 10
;answer
;; (defn sum [v]  (reduce + v))
;tests
;; (ns vectors-choose-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [sum]]))
;; (assert-solution
;;  [[[]] [[1 2 3 4]] [[10 -20]]] [0 10 -10] sum)

;--30)
;Реализуйте функцию, которая разделит вектор на конкретное количество частей.
;Примеры:
;(partiphify [1] 2); => [[1] []]
;(partiphify [1 2 3] 3) ; => [[1] [2] [3]]
;(partiphify [1 2 3 4 5] 2); => [[1 2 3] [4 5]]
;doesn't work
;; (defn partiphify [v n]
;;   (def c (count v))
;;   (def pv (/ c n))
;;   (defn f [v counter] (cond 
;;                         (= c counter)  (into [] (repeat (- n c 1) []))
;;                         (= 0 counter) (conj [] (vector (first v)) (f (into [] (rest v)) (inc counter)))
;;                         (> n counter) (conj (vector (first v)) (f (into [] (rest v)) (inc counter)))
;;                                      ;(= n counter) []
;;                         :else (into [] (rest v))))
;;   (f v 0))
;doesn't work
;; (defn partiphify [v n]
;;   (def c (count v))
;;   (def pv (/ c n))
;;   (defn f [v counter] (cond
;;                         (= c counter) (repeat (- n c 1) [])
;;                         (= 0 counter) (conj (vector (first v)) (f (into [] (rest v)) (inc counter)))
;;                         (> n counter) (conj (vector (first v)) (f (into [] (rest v)) (inc counter)))
;;                                      ;(= n counter) []
;;                         :else (into [] (rest v))))
;;   (f v 0))
;doesnt work
;; (defn partiphify [v n]
;;   (def c (count v))
;;   (def pv (/ c n))
;;   (defn f [r v m] (cond                        
;;                         (empty? r) (f (vector(vector  (first v))) (into [] (rest v)) (dec m))
;;                         (empty? v) (vector (repeat (- n c 1) []))
;;                         (> m 0)  (f (conj r (vector (first v)) (into [] (rest v)) (dec m)))                                   
;;                         :else (conj r (into [] (rest v)))))
;;   (f [] v 0) )
;works
(defn partiphify [v n]
  (def c (count v))
  (def pv (/ c n))
  (defn f [r v m] (cond
                    (empty? r) (f (vector (vector  (first v))) (into [] (rest v)) (dec m))
                    (and (empty? v) (<= 0 (- n c)))(into [](concat r (repeat (- n c) [])))
                    (= m 1)  (f (conj r  v)  [] (dec m))
                    (> m 1)  (f (conj r (vector (first v))) (into [] (rest v)) (dec m))
                    :else  r ))
  (f [] v n))
(partiphify [1] 2); => [[1] []]
(partiphify [1 2 3] 3) ; => [[1] [2] [3]]
(partiphify [1 2 3 4 5] 2); => [[1] [2 3 4 5]] 
;tests
;; (ns immutable-structures-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [partiphify]]))
;; (assert-solution
;;  [[[1] 2] [[1 2 3] 3] [[1 2 3 4 5] 2]]
;;  [[[1] []] [[1] [2] [3]] [[1 2 3] [4 5]]]
;;  partiphify)
(defn partiphify [v n]
  (def c (count v))
  (def p (/ c n))    
  (if (< 0 (- n c)) (into [](concat (vector v) (repeat (- n c) []))) 
      (->> (partition-all p v)(map vec) vec)))
(partiphify [1] 2); => [[1] []]
(partiphify [1 2 3] 3) ; => [[1] [2] [3]]
(partiphify [1 2 3 4 5] 2); => [[1 2 3] [4 5]](conj [9 0] [0] 7)
;answer
;; (defn partiphify [numbers parts]
;;   (let [part (int (Math/ceil (/ (count numbers) parts)))
;;         divided-vec (vec (map vec (partition-all part numbers)))
;;         final-vec (if (not= (count divided-vec) parts) (conj divided-vec []) divided-vec)]
;;     final-vec))
;; (println p)

;--31)
;Реализуйте функцию my-contains?, которая проверяет, есть ли переданный элемент в коллекции. 
;Для конвертации nil можно воспользоваться функцией boolean. Метод .contains использовать нельзя :)
;Примеры:
;(my-contains? [1 2 4 9] 2) ; => true
;(my-contains? [1 2 4 9] 0) ; => false
;(my-contains? [1 2 4 9] 9) ; => true
(defn my-contains? [coll x] (->> (some (hash-set x) coll) (boolean)))
(my-contains? [1 2 4 9] 2) ; => true
(my-contains? [1 2 4 9] 0) ; => false
(my-contains? [1 2 4 9] 9) ; => true
;tests
;; (ns contains-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [my-contains?]]))
;; (assert-solution
;;  [[[1 2 4 9] 2] [[1 2 4 9] 0] [[1 2 4 9] 9]]
;;  [true false true]
;;  my-contains?)
;answer
;; (defn my-contains? [coll elem]  (boolean (some #(= elem %) coll)))

;--32)
;Создайте функцию resolve, которая достает из хеш-мапы доменов IP, связанный с именем домена. 
;Если такая запись отсутствует, то верните \"DNS_PROBE_FINISHED_NXDOMAIN\".
;Примеры:
;(resolve {"rubyonrails.org" "211.116.107.5" "clojure.org" "103.95.84.1" "phoenixframework.org" "234.214.199.63" "reactjs.org" "20.199.101.214"}
 ; "clojure.org"); => "103.95.84.1"
;(resolve {"rhythm.ru" "201.116.147.4" "building.ru" "103.176.11.27" "hexlet.io" "234.214.199.63" "brass.ru" "201.116.147.4"}
 ; "hexlet.io") ; => "234.214.199.63"
;(resolve {"some.com" "127.0.0.1"} "test.net"); => "DNS_PROBE_FINISHED_NXDOMAIN"
(defn resolve [coll x] (if  (nil? (some coll (vector x))) "DNS_PROBE_FINISHED_NXDOMAIN" (some coll (vector x))))
(resolve {"rubyonrails.org" "211.116.107.5" "clojure.org" "103.95.84.1" "phoenixframework.org" "234.214.199.63" "reactjs.org" "20.199.101.214"}
         "clojure.org"); => "103.95.84.1"
(resolve {"rhythm.ru" "201.116.147.4" "building.ru" "103.176.11.27" "hexlet.io" "234.214.199.63" "brass.ru" "201.116.147.4"}
         "hexlet.io") ; => "234.214.199.63"
(resolve {"some.com" "127.0.0.1"} "test.net"); => "DNS_PROBE_FINISHED_NXDOMAIN" 
;tests
;; (ns intro-hashes-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [resolve]]))
;; (assert-solution
;;  [[{"rubyonrails.org" "211.116.107.5" "clojure.org" "103.95.84.1" "phoenixframework.org" "234.214.199.63" "reactjs.org" "20.199.101.214"} "clojure.org"]
;;   [{"rhythm.ru" "201.116.147.4" "building.ru" "103.176.11.27" "hexlet.io" "234.214.199.63" "brass.ru" "201.116.147.4"} "hexlet.io"]
;;   [{"some.com" "127.0.0.1"} "test.net"]]
;;  ["103.95.84.1" "234.214.199.63" "DNS_PROBE_FINISHED_NXDOMAIN"]
;;  resolve)
;answer
;; (defn resolve [domains domain]  (get domains domain "DNS_PROBE_FINISHED_NXDOMAIN"))

;--33)
;Создайте функцию freq, которая принимает коллекцию элементов и создает хеш-мапу с частотой появления элементов в коллекции. Больше подробностей в примерах.
;; (freq ["a" "b" "c" "a" "a" "c" "a" "d" "b"]); => {"a" 4, "b" 2, "c" 2, "d" 1}
;; (freq []); => {}
;; (freq ["Clojure" "Ruby" "Clojure" "Elixir" "Ruby" "HTML" "JS"]); => {"Clojure" 2, "Ruby" 2, "Elixir" 1, "HTML" 1, "JS" 1}
;; (freq [10 10 10 20 300 41 53]); => {10 3, 20 1, 300 1, 41 1, 53 1}
;; (freq [:a :b :c :d :a :a]); => {:a 3, :b 1, :c 1, :d 1}
;variant 1 - works
(defn freq [coll] (merge {} (zipmap coll (map #(count (filter #{%} coll)) coll))))
;variant 2 - works
(defn freq [coll] ( frequencies coll))
(freq ["a" "b" "c" "a" "a" "c" "a" "d" "b"]); => {"a" 4, "b" 2, "c" 2, "d" 1}
(freq []); => {}
(freq ["Clojure" "Ruby" "Clojure" "Elixir" "Ruby" "HTML" "JS"]); => {"Clojure" 2, "Ruby" 2, "Elixir" 1, "HTML" 1, "JS" 1}
(freq [10 10 10 20 300 41 53]); => {10 3, 20 1, 300 1, 41 1, 53 1}
(freq [:a :b :c :d :a :a]); => {:a 3, :b 1, :c 1, :d 1}
; tests
;; (ns intro-hashes-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [freq]]))
;; (assert-solution
;;  [[["a" "b" "c" "a" "a" "c" "a" "d" "b"]]
;;   [[]]
;;   [["Clojure" "Ruby" "Clojure" "Elixir" "Ruby" "HTML" "JS"]]
;;   [[10 10 10 20 300 41 53]]
;;   [[:a :b :c :d :a :a]]]
;;  [{"a" 4 "b" 2 "c" 2 "d" 1}
;;   {}
;;   {"Clojure" 2 "Ruby" 2 "Elixir" 1 "HTML" 1 "JS" 1}
;;   {10 3 20 1 300 1 41 1 53 1}
;;   {:a 3 :b 1 :c 1 :d 1}]
;;  freq)

;-34)
;Реализуйте функцию transit, которая принимает два атома, которые представляют счета в банках и число денег, 
;которое нужно перевести с первого на второй аккаунт, в результате выполнения функции, верните счета в виде вектора. 
;Больше подробностей в примерах.
;(transit (atom 100) (atom 20) 20); => [80 40]
;(transit (atom 50) (atom 30) 50); => [0 80]
(defn transit [a b n] [(swap! a - n) (swap! b + n)])
(transit (atom 100) (atom 20) 20); => [80 40]
(transit (atom 50) (atom 30) 50); => [0 80]
;tests
;; (ns about-state-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [transit]]))
;; (assert-solution
;;  [[(atom 100) (atom 50) 20] [(atom 10) (atom 100) 10] [(atom 50) (atom 30) 50]]
;;  [[80 70] [0 110] [0 80]]
;;  transit)
;answer 
;; (defn transit [first-acc second-acc amount]
;;   (let [first-proceeded (swap! first-acc - amount)
;;         second-proceeded (swap! second-acc + amount)]
;;     [first-proceeded second-proceeded]))

;--35)
;Реализуйте функцию-валидатор vec-even, которая проверяет, что атом является вектором 
;и все его элементы четные (пустой вектор не является валидным случаем) .
;(vec-even [])        ; => false
;(vec-even [0 2 4 6]) ; => true
;(vec-even [1 3 5])   ; => false
(defn vec-even? [v](if 
                   (and  
                     (vector? v) 
                     (not (empty? v)) 
                     (every? even? v) )
                   true false))
(vec-even? [])        ; => false
(vec-even? [0 2 4 6]) ; => true
(vec-even? [1 3 5])   ; => false
;tests
;; (ns atoms-validation-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [vec-even?]]))
;; (assert-solution
;;  [[[0 2 4 6]] [[1 3 5]] [[]] [[0 2 4 5]] [[2]]]
;;  [true false false false true]
;;  vec-even?)
;answer
;; (defn vec-even? [avec]
;;   (and
;;    (not (empty? avec))
;;    (vector? avec)
;;    (= (count avec) (count (filterv even? avec)))))

;--36)
;Реализуйте функцию transit, которая ведет себя так же, как в упражении с атомами, только с помощью агентов. 
;Функция принимает два агента, которые представляют счета в банках и число денег, 
;которое нужно перевести с первого на второй аккаунт, в результате выполнения функции, 
;верните счета в виде вектора (помните, изменения в агентах применяются асинхронно!) .
;(transit (agent 100) (agent 20) 20); => [80 40]
;(transit (agent 50) (agent 30) 50); => [0 80]
(defn transit [a b n] (await (send a - n)) (await (send b + n))[@a @b])
(transit(agent 100) (agent 20) 20); => [80 40]
(transit (agent 50) (agent 30) 50); => [0 80]
;tests
;; (ns agents-test
;;   (:require [test-helper :refer [assert-solution]]
;;             [index :refer [transit]]))
;; (assert-solution
;;  [[(agent 100) (agent 50) 20] [(agent 10) (agent 100) 10] [(agent 50) (agent 30) 50]]
;;  [[80 70] [0 110] [0 80]]
;;  transit)

;--37)
;Создайте атом (начальное значение 0) и добавьте к нему наблюдателя, который при изменении 
;атома выводит сообщение (с помощью print) в следующем виде Change state from x to y., 
;где x - прошлое состояние атома, а y - новое состояние атома. Затем дважды увеличьте атом на 1, 
;потом уменьшите его значение на 1.
(def atom_1 (atom 0))
(add-watch atom_1 "watcher_1" (fn [key variable old newp] 
                                (def ss (str "Change state from " old " to " newp "."))
                                (print ss))) 
(swap! atom_1 + 1)
(swap! atom_1 + 1)
(swap! atom_1 - 1)
;tests
;; (ns watchers-test
;;   (:require [test-helper :refer [assert-output]]))
;; (assert-output "Change state from 0 to 1.Change state from 1 to 2.Change state from 2 to 1.")
;answer
;; (def my-atom (atom 0))
;; (add-watch my-atom :watcher
;;            (fn [_ _ old new]
;;              (print (str "Change state from " old " to " new "."))))
;; (swap! my-atom inc)
;; (swap! my-atom inc)
;; (swap! my-atom dec)

;--38)
;Курс находится в разработке!
;tests
;; (ns intro-polymorphism-test
;;   (:require [test-helper :refer [assert-solution]]))
;; (assert-solution [[1]] [1] identity)

;--39)
;Курс находится в разработке!

;--40)
;Курс находится в разработке!