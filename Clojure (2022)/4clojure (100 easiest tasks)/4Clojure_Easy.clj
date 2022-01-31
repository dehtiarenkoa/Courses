;--19.
;Last Element
;; Write a function which returns the last element in a sequence.
;;     (= (__ [1 2 3 4 5]) 5)
;;     (= (__ '(5 4 3)) 3)
;;     (= (__ ["b" "c" "d"]) "d")
;; Special Restrictions : last
(fn foo [v] (if (second v) (foo (rest v)) (first v)))

;--20.
;Penultimate Element
;; Write a function which returns the second to last element from a sequence.
;; (= (__ (list 1 2 3 4 5)) 4)
;; (= (__ ["a" "b" "c"]) "b")
;; (= (__ [[1 2] [3 4]]) [1 2])
(fn foo [v] (if (second (rest v)) (foo (rest v)) (first v)))

;--21.
;Nth Element
;; Write a function which returns the Nth element from a sequence.
;;     (= (__ '(4 5 6 7) 2) 6)
;;     (= (__ [:a :b :c] 0) :a)
;;     (= (__ [1 2 3 4] 1) 2)
;;     (= (__ '([1 2] [3 4] [5 6]) 2) [5 6])
;; Special Restrictions : nth
(fn foo [v n] (if (= n 0) (first v) (foo (rest v) (dec n))))

;--22.
;Count a Sequence
;; Write a function which returns the total number of elements in a sequence.
;;     (= (__ '(1 2 3 3 1)) 5)
;;     (= (__ "Hello World") 11)
;;     (= (__ [[1 2] [3 4] [5 6]]) 3)
;;     (= (__ '(13)) 1)
;;     (= (__ '(:a :b :c)) 3)
;; Special Restrictions : count
(fn foo [v]  (+ (if (empty? v)  0 (+ 1 (foo (rest v))))))

;--23.
;Reverse a Sequence
;; Write a function which reverses a sequence.
;;     (= (__ [1 2 3 4 5]) [5 4 3 2 1])
;;     (= (__ (sorted-set 5 7 2 7)) '(7 5 2))
;;     (= (__ [[1 2][3 4][5 6]]) [[5 6][3 4][1 2]])
;; Special Restrictions : reverse
(fn foo [v] (if (not (empty? v)) (cons (last v) (foo (butlast v)))))

;--24.
;Sum It All Up
;; Write a function which returns the sum of a sequence of numbers.
;; (= (__ [1 2 3]) 6)
;; (= (__ (list 0 -2 5 5)) 8)
;; (= (__ #{4 2 1}) 7)
;; (= (__ '(0 0 -1)) -1)
;; (= (__ '(1 10 3)) 14)
#(reduce + %)

;--25.
;Find the odd numbers
;; Write a function which returns only the odd numbers from a sequence.
;; (= (__ #{1 2 3 4 5}) '(1 3 5))
;; (= (__ [4 2 1 6]) '(1))
;; (= (__ [2 2 4 6]) '())
;; (= (__ [1 1 1 3]) '(1 1 1 3))
#(filter odd? %)

;--26.
;Fibonacci Sequence
;; Write a function which returns the first X fibonacci numbers.
;; (= (__ 3) '(1 1 2))
;; (= (__ 6) '(1 1 2 3 5 8))
;; (= (__ 8) '(1 1 2 3 5 8 13 21))
; from  Miller's "Programming Clojure" book
(fn [n] (def fib (map first (iterate (fn [[a b]] [b (+ a b)]) [1 1]))) (take n fib))

;--27.
;Palindrome Detector
;; Write a function which returns true if the given sequence is a palindrome. Hint: "racecar" does not equal '(\r \a \c \e \c \a \r)
;;     (false? (__ '(1 2 3 4 5)))
;;     (true? (__ "racecar"))
;;     (true? (__ [:foo :bar :foo]))
;;     (true? (__ '(1 1 3 3 1 1)))
;;     (false? (__ '(:a :b :c)))
(fn foo [s] (= (reverse s) (seq s)))

;--28.
;Flatten a Sequence
;; Write a function which flattens a sequence.
;;     (= (__ '((1 2) 3 [4 [5 6]])) '(1 2 3 4 5 6))
;;     (= (__ ["a" ["b"] "c"]) '("a" "b" "c"))
;;     (= (__ '((((:a))))) '(:a))
;; Special Restrictions : flatten
(fn [x] (filter (complement sequential?) (rest (tree-seq sequential? seq x))))
(fn [x] (filter (complement sequential?) (rest (tree-seq sequential? identity x))))


;--29.
;Get the Caps
;; Write a function which takes a string and returns a new string containing only the capital letters.
;; (= (__ "HeLlO, WoRlD!") "HLOWRD")
;; (empty? (__ "nothing"))
;; (= (__ "$#A(*&987Zf") "AZ")
(fn [s]   (clojure.string/join (re-seq  #"[A-Z]" s)))

;--30.
;Compress a Sequence
;; Write a function which removes consecutive duplicates from a sequence.
;; (= (apply str (__ "Leeeeeerrroyyy")) "Leroy")
;; (= (__ [1 1 2 3 3 2 2 3]) '(1 2 3 2 3))
;; (= (__ [[1 2] [1 2] [3 4] [1 2]]) '([1 2] [3 4] [1 2]))
dedupe

;--31.
;Pack a Sequence
;; Write a function which packs consecutive duplicates into sub-lists.
;; (= (__ [1 1 2 1 1 1 3 3]) '((1 1) (2) (1 1 1) (3 3)))
;; (= (__ [:a :a :b :b :c]) '((:a :a) (:b :b) (:c)))
;; (= (__ [[1 2] [1 2] [3 4]]) '(([1 2] [1 2]) ([3 4])))
#(partition-by identity %)

;--32.
;Duplicate a Sequence
;; Write a function which duplicates each element of a sequence.
;; (= (__ [1 2 3]) '(1 1 2 2 3 3))
;; (= (__ [:a :a :b :b]) '(:a :a :a :a :b :b :b :b))
;; (= (__ [[1 2] [3 4]]) '([1 2] [1 2] [3 4] [3 4]))
;; (= (__ [44 33]) [44 44 33 33])
#(mapcat (fn [el] (repeat 2 el)) %)

;--33.
;Replicate a Sequence
;; Write a function which replicates each element of a sequence a variable number of times.
;; (= (__ [1 2 3] 2) '(1 1 2 2 3 3))
;; (= (__ [:a :b] 4) '(:a :a :a :a :b :b :b :b))
;; (= (__ [4 5 6] 1) '(4 5 6))
;; (= (__ [[1 2] [3 4]] 2) '([1 2] [1 2] [3 4] [3 4]))
;; (= (__ [44 33] 2) [44 44 33 33])
#(mapcat (fn [el] (repeat %2 el)) %1)

;--34
;Implement range
;; Write a function which creates a list of all integers in a given range.
;;     (= (__ 1 4) '(1 2 3))
;;     (= (__ -2 2) '(-2 -1 0 1))
;;     (= (__ 5 8) '(5 6 7))
;; Special Restrictions : range
(fn [a b] (letfn [(incc [x] (if (< x b) (cons x (incc (inc x))) ()))] (incc a)))

;--38.
;Maximum value
;; Write a function which takes a variable number of parameters and returns the maximum value.
;;     (= (__ 1 8 3 4) 8)
;;     (= (__ 30 20) 30)
;;     (= (__ 45 67 11) 67)
;; Special Restrictions : max,max-key
(fn [& s]  (letfn [(r [m coll]  (if (empty? coll)
                                  m
                                  (r (if (> (first coll) m) (first coll) m) (rest coll))))]
             (r (first s) (rest s))))

;--39.
;Interleave Two Seqs
;; Write a function which takes two sequences and returns the first item from each, then the second item from each, then the third, etc.
;;     (= (__ [1 2 3] [:a :b :c]) '(1 :a 2 :b 3 :c))
;;     (= (__ [1 2] [3 4 5 6]) '(1 3 2 4))
;;     (= (__ [1 2 3 4] [5]) [1 5])
;;     (= (__ [30 20] [25 15]) [30 25 20 15])
;; Special Restrictions : interleave
;works, with saving the type of the initial collection
(fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                     (rec (conj res (first x) (first y)) (rest x) (rest y))
                                     res))]
              (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy))))
;realisation of the "interleave" from clojure.core:
;; (defn interleave
;;   "Returns a lazy seq of the first item in each coll, then the second etc."
;;   {:added "1.0"
;;    :static true}
;;   ([] ())
;;   ([c1] (lazy-seq c1))
;;   ([c1 c2]
;;    (lazy-seq
;;     (let [s1 (seq c1) s2 (seq c2)]
;;       (when (and s1 s2)
;;         (cons (first s1) (cons (first s2)
;;                                (interleave (rest s1) (rest s2))))))))
;;   ([c1 c2 & colls]
;;    (lazy-seq
;;     (let [ss (map seq (conj colls c2 c1))]
;;       (when (every? identity ss)
;;         (concat (map first ss) (apply interleave (map rest ss))))))))

;--40.
;Interpose a Seq
;; Write a function which separates the items of a sequence by an arbitrary value.
;;     (= (__ 0 [1 2 3]) [1 0 2 0 3])
;;     (= (apply str (__ ", " ["one" "two" "three"])) "one, two, three")
;;     (= (__ :z [:a :b :c :d]) [:a :z :b :z :c :z :d])
;; Special Restrictions : interpose
;works, but long
(fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (reduce concat (map #(conj (empty sx) % sep) sx))))
;works
(fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (mapcat #(conj (empty sx) % sep) sx)))

;--41.
;Drop Every Nth Item
;; Write a function which drops every Nth item from a sequence.
;; (= (__ [1 2 3 4 5 6 7 8] 3) [1 2 4 5 7 8])
;; (= (__ [:a :b :c :d :e :f] 2) [:a :c :e])
;; (= (__ [1 2 3 4 5 6] 4) [1 2 3 5 6])
(fn [s n]
  (letfn [(foo [r res sx] (if (not-empty sx)
                            (if (not= 0 (first r))
                              (foo (rest r) (conj res (first sx)) (rest sx))
                              (foo (rest r) res  (rest sx)))
                            res))]
    (foo (cycle (reverse (range n))) (empty s) s)))

;--42.
;Factorial Fun
;; Write a function which calculates factorials.
;; (= (__ 1) 1)
;; (= (__ 3) 6)
;; (= (__ 5) 120)
;; (= (__ 8) 40320)
(defn fac
  ([n] (cond
         (= n 1) 1
         (> n 1) (* n (fac (dec n))))))
;(letfn [(fac [n](while (> n 0) (* n (fac (dec n)))) )](fac n))

;--45.
;Intro to Iterate
;; The iterate function can be used to produce an infinite lazy sequence.
;; (= __ (take 5 (iterate #(+ 3 %) 1)))
'(1 4 7 10 13)

;--47.
;Contain Yourself
;; The contains? function checks if a KEY is present in a given collection. 
;; This often leads beginner clojurians to use it incorrectly with numerically indexed collections like vectors and lists.
;; (contains? #{4 5 6} __)
;; (contains? [1 1 1 1 1] __)
;; (contains? {4 :a 2 :b} __)
;; (not (contains? [1 2 4] __))
4

;--48.
;Intro to some
;; The some function takes a predicate function and a collection. 
;; It returns the first logical true value of (predicate x) where x is an item in the collection.
;; (= __ (some #{2 7 6} [5 6 7 8]))
;; (= __ (some #(when (even? %) %) [5 6 7 8]))
6

;--49.
;Split a sequence
;; Write a function which will split a sequence into two parts.
;;     (= (__ 3 [1 2 3 4 5 6]) [[1 2 3] [4 5 6]])
;;     (= (__ 1 [:a :b :c :d]) [[:a] [:b :c :d]])
;;     (= (__ 2 [[1 2] [3 4] [5 6]]) [[[1 2] [3 4]] [[5 6]]])
;; Special Restrictions : split-at
(fn [n s]  (vector
            (into [] (drop-last (- (count s) n) s))
            (into [] (drop n s))))

;--51.
;Advanced Destructuring
;; Here is an example of some more sophisticated destructuring.
;; (= [1 2 [3 4 5] [1 2 3 4 5]] (let [[a b & c :as d] __] [a b c d]))
[1 2 3 4 5]

;--61.
;Map Construction
;; Write a function which takes a vector of keys and a vector of values and constructs a map from them.
;;     (= (__ [:a :b :c] [1 2 3]) {:a 1, :b 2, :c 3})
;;     (= (__ [1 2 3 4] ["one" "two" "three"]) {1 "one", 2 "two", 3 "three"})
;;     (= (__ [:foo :bar] ["foo" "bar" "baz"]) {:foo "foo", :bar "bar"})
;; Special Restrictions : zipmap
(fn [k v] (apply hash-map (reverse (interleave v k))))

;--62.
;Re-implement Iteration
;; Given a side-effect free function f and an initial value x write a function which returns 
;; an infinite lazy sequence of x, (f x), (f (f x)), (f (f (f x))), etc.
;;     (= (take 5 (__ #(* 2 %) 1)) [1 2 4 8 16])
;;     (= (take 100 (__ inc 0)) (take 100 (range)))
;;     (= (take 9 (__ #(inc (mod % 3)) 1)) (take 9 (cycle [1 2 3])))
;; Special Restrictions : iterate
; from Miller's "Programming Clojure"
;; (defn lazy-seq-fibo
;;   ([]
;;    (concat [0 1] (lazy-seq-fibo 0N 1N)))
;;   ([a b]
;;    (let [n (+ a b)]
;;      (lazy-seq
;;       (cons n (lazy-seq-fibo b n))))))
(fn [f x] ((fn lazy
             ([]
              (concat [x] (lazy x)))
             ([x]
              (let [n (f x)]
                (lazy-seq
                 (cons n (lazy n))))))))

;--63.
;Group a Sequence
;; Given a function f and a sequence s, write a function which returns a map. The keys should be the values 
;; of f applied to each item in s. The value at each key should be a vector of corresponding items in the order they appear in s.
;; (= (__ #(> % 5) #{1 3 6 8}) {false [1 3], true [6 8]})
;; (= (__ #(apply / %) [[1 2] [2 4] [4 6] [3 6]])
;;        {1/2 [[1 2] [2 4] [3 6]], 2/3 [[4 6]]})
;; (= (__ count [[1] [1 2] [3] [1 2 3] [2 3]])
;;        {1 [[1] [3]], 2 [[1 2] [2 3]], 3 [[1 2 3]]})
;; Special Restrictions : group-by
(fn [f s]
  (let [k (map f s)]
    (apply merge-with into (map (fn [a b] {a [b]}) k s))))

;--66.
;Greatest Common Divisor
;; Given two integers, write a function which returns the greatest common divisor.
;; (= (__ 2 4) 2)
;; (= (__ 10 5) 5)
;; (= (__ 5 7) 1)
;; (= (__ 1023 858) 33)

(fn [a b] (letfn [(foo [a b x]
                    (if (and (= (rem a x) 0)
                             (= (rem b x) 0))
                      x
                      (foo a b (dec x))))]
            (foo a b (min a b))))




;--81.
;Set Intersection
;; Write a function which returns the intersection of two sets. 
;; The intersection is the sub-set of items that each set has in common.
;;     (= (__ #{0 1 2 3} #{2 3 4 5}) #{2 3})
;;     (= (__ #{0 1 2} #{3 4 5}) #{})
;;     (= (__ #{:a :b :c :d} #{:c :e :a :f :d}) #{:a :c :d})
;; Special Restrictions : intersection
(fn [a b]
  (set (filter #(not (nil? %)) (map a b))))

;--83.
;	Set Intersection
;; Write a function which takes a variable number of booleans. 
;; Your function should return true if some of the parameters are true, 
;; but not all of the parameters are true. Otherwise your function should return false.
;; (= false (__ false false))
;; (= true (__ true false))
;; (= false (__ true))
;; (= true (__ false true false))
;; (= false (__ true true true))
;; (= true (__ true true true false))
(fn [& s]
  (if (and
       (reduce (fn [a b] (or a b)) s)
       (not (reduce (fn [a b] (and a b)) s)))
    true
    false))

;--88.
;Symmetric Difference
;; Write a function which returns the symmetric difference of two sets. 
;; The symmetric difference is the set of items belonging to one but not both of the two sets.
;; (= (__ #{1 2 3 4 5 6} #{1 3 5 7}) #{2 4 6 7})
;; (= (__ #{:a :b :c} #{}) #{:a :b :c})
;; (= (__ #{} #{4 5 6}) #{4 5 6})
;; (= (__ #{[1 2] [2 3]} #{[2 3] [3 4]}) #{[1 2] [3 4]})
(fn [a b] (set (map first (filter #(= (second %) 1) (frequencies (concat a b))))))

;--90.
;Cartesian product
;; Write a function which calculates the Cartesian product of two sets.
;; (= (__ #{"ace" "king" "queen"} #{"♠" "♥" "♦" "♣"})
;;    #{["ace"   "♠"] ["ace"   "♥"] ["ace"   "♦"] ["ace"   "♣"]
;;      ["king"  "♠"] ["king"  "♥"] ["king"  "♦"] ["king"  "♣"]
;;      ["queen" "♠"] ["queen" "♥"] ["queen" "♦"] ["queen" "♣"]})
;; (= (__ #{1 2 3} #{4 5})
;;    #{[1 4] [2 4] [3 4] [1 5] [2 5] [3 5]})
;; (= 300 (count (__ (into #{} (range 10))
;;                   (into #{} (range 30)))))
(fn [a b] (set (for [x a
                     y b]
                 [x y])))

;--95.
;To Tree, or not to Tree
;; Write a predicate which checks whether or not a given sequence represents a binary tree. 
;; Each node in the tree must have a value, a left child, and a right child.
;; (= (__ '(:a (:b nil nil) nil))   true)
;; (= (__ '(:a (:b nil nil)))   false)
;; (= (__ [1 nil [2 [3 nil nil] [4 nil nil]]])   true)
;; (= (__ [1 [2 nil nil] [3 nil nil] [4 nil nil]])   false)
;; (= (__ [1 [2 [3 [4 nil nil] nil] nil] nil])   true)
;; (= (__ [1 [2 [3 [4 false nil] nil] nil] nil])   false)
;; (= (__ '(:a nil ()))   false)
(fn [ss] (letfn [(is-tree? [s] (and
                                (sequential? s)
                                (boolean (not-empty s))
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (#(or (nil? %) (is-tree? %)) (second s))
                                (#(or (nil? %)  (is-tree? %)) (last s))))]
           (is-tree? ss)))

;--96.
;Beauty is Symmetry
;; Let us define a binary tree as "symmetric" if the left half of the tree is the mirror 
;; image of the right half of the tree. Write a predicate to determine whether or not a 
;; given binary tree is symmetric. 
;; (see To Tree, or not to Tree for a reminder on the tree representation we're using) .
;; (= (__ '(:a (:b nil nil) (:b nil nil))) true)
;; (= (__ '(:a (:b nil nil) nil)) false)
;; (= (__ '(:a (:b nil nil) (:c nil nil))) false)
;; (= (__ [1 [2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]]
;;         [2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil]])
;;    true)
;; (= (__ [1 [2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]]
;;         [2 [3 nil [4 [5 nil nil] [6 nil nil]]] nil]])
;;    false)
;; (= (__ [1 [2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]]
;;         [2 [3 nil [4 [6 nil nil] nil]] nil]])
;;    false)
(fn [ss] (letfn [(is-tree? [s] (and
                                (sequential? s)
                                (boolean (not-empty s))
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (#(or (nil? %) (is-tree? %)) (second s))
                                (#(or (nil? %)  (is-tree? %)) (last s))))
                 (firsts [s] (if (sequential? s) (first s) s))
                 (rev [s] (if (sequential? s) (cons (first s) (reverse (rest s))) s))]
           (and
            (is-tree? ss)
            (= (map firsts (tree-seq sequential? identity (second ss)))
               (map firsts (tree-seq sequential? rev (last ss)))))))

;--97.
;Pascal's Triangle
;; Pascal's triangle is a triangle of numbers computed using the following rules: 
;; - The first row is 1. 
;; - Each successive row is computed by adding together adjacent numbers in the row above, 
;; and adding a 1 to the beginning and end of the row. 
;; Write a function which returns :the nth row of Pascal's Triangle.
;;     (= (__ 1) [1])
;;     (= (map __ (range 1 6))
;;        [     [1]
;;             [1 1]
;;            [1 2 1]
;;           [1 3 3 1]
;;          [1 4 6 4 1]])
;;     (= (__ 11)
;;        [1 10 45 120 210 252 210 120 45 10 1])
;;     (= (__ 11)
;;        [1 10 45 120 210 252 210 120 45 10 1])
(fn ([n] (letfn [(foo [prev m]                     
                    (if (= m n)
                      prev
                      (foo (concat '(1) (red prev) '(1)) (inc m))))
                  (red [s] (map #(apply + %)
                                (partition 2
                                           (concat
                                            (list (first s))
                                            (mapcat #(repeat 2 %) (rest (butlast s)))
                                            (list (last s))))))]
            (case n
              1 [1]
              2 [1 1]
             (into [] (foo '(1) 2))))))

;--99.
;Product Digits
;; Write a function which multiplies two numbers and returns the result as a sequence of its digits.
;; (= (__ 1 1) [1])
;; (= (__ 99 9) [8 9 1])
;; (= (__ 999 99) [9 8 9 0 1])
(fn [a b] (mapv #(-> % (str) (read-string)) (seq (str (* a b)))))
;or (fn [a b] (mapv #(-> % (str) (Integer/parseInt)) (seq (str (* a b)))))


;--100.
;Least Common Multiple
;; Write a function which calculates the least common multiple. 
;; Your function should accept a variable number of positive integers or ratios.
;; (== (__ 2 3) 6)
;; (== (__ 5 3 7) 105)
;; (== (__ 1/3 2/5) 2)
;; (== (__ 3/4 1/6) 3/2)
;; (== (__ 7 5/7 2 3/5) 210)
(fn [& ss] (letfn
                 [(r-even? [n] (if (ratio? n)
                                 (and (even? (numerator n)) (even? (denominator n)))
                                 (even? n)))
                  (r-odd? [n] (complement r-even?))
                  (nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
                    (cond
                      (= a 0) b
                      (= b 0) b
                      (= a b) a
                      (and (r-even? a) (r-even? b)) (* 2 (nod (/ a 2) (/ b 2)))
                      (and (r-even? a) (r-odd? b)) (nod (/ a 2) b)
                      (and (r-even? b) (r-odd? a)) (nod (/ b 2) a)
                      (and (r-odd? b) (r-odd? a) (> a b)) (nod (/ (- a b) 2) b)
                      (and (r-odd? b) (r-odd? a) (< a b)) (nod (/ (- b a) 2) a)))
                  (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
                    (/ (* a b) (nod a b)))]
                  (if (some ratio? ss)
                    (let [denominators-nok (reduce nok (map #(if (ratio? %) (denominator %) 1) ss))
                          numerators (map #(* (if (ratio? %) (numerator %) %) 
                                              (/ denominators-nok 
                                                 (if (ratio? %) (denominator %) 1))) ss)
                          numerators-nok (reduce nok numerators)]
                      (rationalize (/ numerators-nok denominators-nok)))
                    (reduce nok ss))))


;--107.
;Simple closures
;; Lexical scope and first-class functions are two of the most basic building blocks of a functional language like Clojure.
;; When you combine the two together, you get something very powerful called lexical closures. 
;; With these, you can exercise a great deal of control over the lifetime of your local bindings, 
;; saving their values for use later, long after the code you're running now has finished. 
;; It can be hard to follow in the abstract, so let's build a simple closure. Given a positive integer n, 
;; return a function (f x) which computes xn. Observe that the effect of this is to preserve the value 
;; of n for use outside the scope in which it is defined.
;; (= 256 ((__ 2) 16), ((__ 8) 2))
;; (= [1 8 27 64] (map (__ 3) [1 2 3 4]))
;; (= [1 2 4 8 16] (map #((__ %) 2) [0 1 2 3 4]))
(fn [n] (fn [x] (int (reduce * (repeat n x))))) ; or (fn [n] (fn [x] (int (Math/pow x n))))

;--118.
;Re-implement Map
;; Map is one of the core elements of a functional programming language. 
;; Given a function f and an input sequence s, return a lazy sequence of (f x) for each element x in s.
;; (= [3 4 5 6 7]
;;    (__ inc [2 3 4 5 6]))
;; (= (repeat 10 nil)
;;    (__ (fn [_] nil) (range 10)))
;; (= [1000000 1000001]
;;    (->> (__ inc (range))
;;         (drop (dec 1000000))
;;         (take 2)))
;; (= [1000000 1000001]
;;    (->> (__ inc (range))
;;         (drop (dec 1000000))
;;         (take 2)))
(fn [f s]
  (lazy-seq
   (if  (seq s)
     (cons
      (f (first s))
      (map (rest s))))))


;--120.
;Sum of square of digits
;; Write a function which takes a collection of integers as an argument. 
;; Return the count of how many elements are smaller than the sum of their squared component digits. 
;; For example: 10 is larger than 1 squared plus 0 squared; whereas 15 is smaller than 1 squared plus 5 squared.
;;     (= 8 (__ (range 10)))
;;     (= 19 (__ (range 30)))
;;     (= 50 (__ (range 100)))
;;     (= 50 (__ (range 1000)))
(fn [s] 
  (letfn 
    [(to-digits [n]
      (->> n str (map (comp read-string str))))
     (sum-of-squares [ss] 
       (reduce + (map #(* % %) ss)))]         
    (count (filter #(< % (sum-of-squares (to-digits %))) s))))

;--122.
;Read a binary number
;; Convert a binary number, provided in the form of a string, to its numerical value.
;; (= 0     (__ "0"))
;; (= 7     (__ "111"))
;; (= 8     (__ "1000"))
;; (= 9     (__ "1001"))
;; (= 255   (__ "11111111"))
;; (= 1365  (__ "10101010101"))
;; (= 65535 (__ "1111111111111111"))
(fn [ss] (let [s (map #(read-string (str %)) ss)
               t (lazy-seq (map #(reduce * (repeat % 2)) (range)))]
           (reduce + (map * (reverse s) t))))

;--126.
;Through the Looking Class
;; Enter a value which satisfies the following:
;;     (let [x __] (and (= (class x) x) x))
java.lang.Class

;--128.
;Recognize Playing Cards
;; A standard American deck of playing cards has four suits - spades, hearts, diamonds, 
;; and clubs - and thirteen cards in each suit. Two is the lowest rank, followed by other integers up to ten; 
;; then the jack, queen, king, and ace. It's convenient for humans to represent these cards as suit/rank pairs, 
;; such as H5 or DQ: the heart five and diamond queen respectively. But these forms are not convenient for programmers, 
;; so to write a card game you need some way to parse an input string into meaningful components. 
;; For purposes of determining rank, we will define the cards to be valued from 0 (the two) to 12 (the ace) 
;; Write a function which converts (for example) the string "SJ" into a map of {:suit :spade,:rank 9}. 
;; A ten will always be represented with the single character "T", rather than the two characters "10".
;; (= {:suit :diamond :rank 10} (__ "DQ"))
;; (= {:suit :heart :rank 3} (__ "H5"))
;; (= {:suit :club :rank 12} (__ "CA"))
;; (= (range 13) (map (comp :rank __ str)
;;                    '[S2 S3 S4 S5 S6 S7
;;                      S8 S9 ST SJ SQ SK SA]))
(fn [s]
  (let [a (str (first s))
        b (str (last s))
        suits {"S" :spade  "H" :heart "D" :diamond "C" :club }
        ranks ["2" "3" "4" "5" "6" "7" "8" "9" "T" "J" "Q" "K" "A"]]
    (letfn 
     [(pos [x coll] 
           (keep-indexed (fn [idx val] (when (= val x)  idx )) coll))]
  (hash-map :suit (last (find suits a)), :rank (first (pos b ranks)))
      )))
  
;--135.
;	Infix Calculator
;; Your friend Joe is always whining about Lisps using the prefix notation for math. 
;; Show him how you could easily write a function that does math using the infix notation. 
;; Is your favorite language that flexible, Joe? Write a function that accepts a variable length 
;; mathematical expression consisting of numbers and the operations +, -, *, and /. 
;; Assume a simple calculator that does not do precedence and instead just calculates left to right.
;;     (= 7  (__ 2 + 5))
;;     (= 42 (__ 38 + 48 - 2 / 2))
;;     (= 8  (__ 10 / 2 - 1 * 2))
;;     (= 72 (__ 20 / 2 + 2 + 4 + 8 - 6 - 10 * 9))
(fn f [a & s]
  (if (second s)
    (recur ((first s) a (second s)) (next (next s)))
    a))

;--143.
;dot product
;; Create a function that computes the dot product of two sequences. You may assume that the vectors will have the same length.
;; (= 0 (__ [0 1 0] [1 0 0]))
;; (= 3 (__ [1 1 1] [1 1 1]))
;; (= 32 (__ [1 2 3] [4 5 6]))
;; (= 256 (__ [2 5 6] [100 10 1]))
(fn [a b]
  (reduce + (map * a b)))

;--146.
;Trees into tables
;; Because Clojure's for macro allows you to "walk" over multiple sequences in a nested fashion, 
;; it is excellent for transforming all sorts of sequences. If you don't want a sequence as your final output 
;; (say you want a map), you are often still best-off using for, because you can produce a sequence and feed it into a map, 
;; for example. For this problem, your goal is to "flatten" a map of hashmaps. Each key in your output map should be the "path" (1) 
;; that you would have to take in the original map to get to a value, so for example {1 {2 3}} should result in {[1 2] 3}. 
;; You only need to flatten one level of maps: if one of the values is a map, just leave it alone. 
;; (1) That is, (get-in original [k1 k2]) should be the same as (get result [k1 k2])
;;     (= (__ '{a {p 1, q 2}
;;                                 b {m 3, n 4}})
;;                           '{[a p] 1, [a q] 2
;;                             [b m] 3, [b n] 4})
;;     (= (__ '{[1] {a b c d}
;;                                 [2] {q r s t u v w x}})
;;                           '{[[1] a] b, [[1] c] d,
;;                             [[2] q] r, [[2] s] t,
;;                             [[2] u] v, [[2] w] x})
;;     (= (__ '{m {1 [a b c] 3 nil}})
;;                           '{[m 1] [a b c], [m 3] nil})
(fn  [ss]  
  (letfn [(foo [res s]      
      (if (and (associative? s) (not (sequential? s)))
        (concat 
         (mapcat 
          #(foo (conj res %1) %2) 
          (keys s) (vals s)))
        (vector res s)))]
    (apply assoc {} (foo [] ss))))

;--147.
;; Write a function that, for any given input vector of numbers, returns an infinite lazy sequence of vectors, 
;; where each next one is constructed from the previous following the rules used in Pascal's Triangle. 
;; For example, for [3 1 2], the next row is [3 4 3 2]. Beware of arithmetic overflow! In clojure (since version 1.3 in 2011), 
;; if you use an arithmetic operator like + and the result is too large to fit into a 64-bit integer, an exception is thrown. 
;; You can use +' to indicate that you would rather overflow into Clojure's slower, arbitrary-precision bigint.
;;     (= (second (__ [2 3 2])) [2 5 5 2])
;;     (= (take 5 (__ [1])) [[1] [1 1] [1 2 1] [1 3 3 1] [1 4 6 4 1]])
;;     (= (take 2 (__ [3 1 2])) [[3 1 2] [3 4 3 2]])
;;     (= (take 100 (__ [2 4 2])) (rest (take 101 (__ [2 2]))))
(fn [vv]
  (letfn [(expand [w] (vec (concat
                            (list (first w))
                            (map #(apply +' %)
                                 (partition 2
                                            (concat
                                             (list (first w))
                                             (mapcat #(repeat 2 %) (rest (butlast w)))
                                             (list (last w)))))
                            (list (last w)))))
          (foo [v]
            (if (= (count v) 1)
              (concat (vector v) (foo  (vector (first v) (last v))))
              (iterate expand  v)))]
    (lazy-seq (foo vv))))

;--153.
;Pairwise Disjoint Sets
;; Given a set of sets, create a function which returns true if no two of those sets have any elements in common (1) 
;; and false otherwise. Some of the test cases are a bit tricky, so pay a little more attention to them. 
;; (1) Such sets are usually called pairwise disjoint or mutually disjoint.
;; (= (__ #{#{\U} #{\s} #{\e \R \E} #{\P \L} #{\.}})
;;        true)
;; (= (__ #{#{:a :b :c :d :e}
;;          #{:a :b :c :d}
;;          #{:a :b :c}
;;          #{:a :b}
;;          #{:a}})
;;    false)
;; (= (__ #{#{[1 2 3] [4 5]}
;;          #{[1 2] [3 4 5]}
;;          #{[1] [2] 3 4 5}
;;          #{1 2 [3 4] [5]}})
;;    true)
;; (= (__ #{#{'a 'b}
;;          #{'c 'd 'e}
;;          #{'f 'g 'h 'i}
;;          #{''a ''c ''f}})
;;    true)
;; (= (__ #{#{'(:x :y :z) '(:x :y) '(:z) '()}
;;          #{#{:x :y :z} #{:x :y} #{:z} #{}}
;;          #{'[:x :y :z] [:x :y] [:z] [] {}}})
;;    false)
;; (= (__ #{#{(= "true") false}
;;          #{:yes :no}
;;          #{(class 1) 0}
;;          #{(symbol "true") 'false}
;;          #{(keyword "yes") ::no}
;;          #{(class '1) (int \0)}})
;;    false)
;; (= (__ (set [(set [distinct?])
;;              (set [#(-> %) #(-> %)])
;;              (set [#(-> %) #(-> %) #(-> %)])
;;              (set [#(-> %) #(-> %) #(-> %)])]))
;;    true)
;; (= (__ #{#{(#(-> *)) + (quote mapcat) #_nil}
;;          #{'+ '* mapcat (comment mapcat)}
;;          #{(do) set contains? nil?}
;;          #{#_empty?}})
;;    false)
(fn [s]
  (let [r (apply concat s)]    
    (=
     (count (set r))
     (count r))))

;--157.
;
;; Transform a sequence into a sequence of pairs containing the original elements along with their index.
;; (= (__ [:a :b :c]) [[:a 0] [:b 1] [:c 2]])
;; (= (__ [0 1 3]) '((0 0) (1 1) (3 2)))
;; (= (__ [[:foo] {:bar :baz}]) [[[:foo] 0] [{:bar :baz} 1]])
(fn [s]
  (keep-indexed #(vector %2 %1) s)
  )

;--166.
;Comparisons
;; For any orderable data type it's possible to derive all of the basic comparison operations (<, ≤, =, ≠, ≥, and >) 
;; from a single operation (any operator but = or ≠ will work). Write a function that takes three arguments, 
;; a less than operator for the data and two items to compare. The function should return a keyword describing 
;; the relationship between the two items. The keywords for the relationship between x and y are as follows: 
;; x = y → :eq x > y → :gt x < y → :lt
;;     (= :gt (__ < 5 1))
;;     (= :eq (__ (fn [x y] (< (count x) (count y))) "pear" "plum"))
;;     (= :lt (__ (fn [x y] (< (mod x 5) (mod y 5))) 21 3))
;;     (= :gt (__ > 0 2))
(fn [f a b]
  (cond
    (f a b) :lt
    (f b a) :gt
    (and
     (not (f b a))
     (not (f a b))) :eq))

;--173.
;Intro to Destructuring 2
;; Sequential destructuring allows you to bind symbols to parts of sequential things (vectors, lists, seqs, etc.): 
;; (let [bindings* ] exprs*) Complete the bindings so all let-parts evaluate to 3.
;;     (= 3
;;        (let [[a] [+ (range 3)]] (apply a))
;;        (let [[[__] b] [[+ 1] 2]] (__ b))
;;        (let [[__] [inc 2]] (__)))
a c
(= 3
  (let [[a c] [+ (range 3)]] (apply a c))
  (let [[[a c] b] [[+ 1] 2]] (a c b))
  (let [[a c] [inc 2]] (a c)))