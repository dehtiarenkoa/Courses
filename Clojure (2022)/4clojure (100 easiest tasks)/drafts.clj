(require '[clojure.string :as str])

(defn all-uppercase? [s]
  (= s (str/upper-case s)))

((fn foo [v] (+ (if (empty v) 0 (+ 1 (foo (rest v))))))  '(1 2 3 3 1))
(if (not-empty []) 1)
(rest [8])
; wrong ((fn foo [v]  (if (not (empty? v)) (+ 1  (if (not (empty? (rest v))) (foo (rest v)) 0))) 0)  '(1 2 3 3 1))
((fn foo [v]  (+ (if (empty? v)  0 (+ 1 (foo (rest v))))))  '(1 2 3 3 1))
;fib number works
((fn fib [n] (letfn [(fb [m]
                       (cond (= m 1) 1
                             (= m 2) 1
                             (> m 2) (+ (fb (- m 1)) (fb (- m 2)))))]
               (fb n))) 6)
;--
(fn fib [n] (conj '() (cond (= m 1) 1
                            (= m 2) 1
                            (> (inc n) m 2))))
(fn fib [n] (let [defn fb [m]
                  (cond (= m 1) 1
                        (= m 2) 1
                        (> m 2) (+ (fb (- m 1)) (fb (- m 2))))]
              (conj '() (fb n))))
((fn fib [n] ((cond
                (= n 1) 1
                (= n 2) 1
                (> n 2) (+ (fib (- n 1)) (fib (- n 2)))))) 3)
(= ((fn fib [n] ((cond
                   (= n 1) 1
                   (= n 2) 1
                   (> n 2) (+ (fib (- n 1)) (fib (- n 2)))))) 3) '(1 1 2))
(= ((fn fib [n] (letfn [(fb [m]
                          (cond (= m 1) 1
                                (= m 2) 1
                                (> m 2) (+ (fb (- m 1)) (fb (- m 2)))))]
                  (conj '() (fb n)))) 6) '(1 1 2 3 5 8))
(= (__ 8) '(1 1 2 3 5 8 13 21))
(conj)

((fn fib [n] (letfn [(fb [m]
                       (cond (= m 1) 1
                             (= m 2) 1
                             (> m 2) (+ (fb (- m 1)) (fb (- m 2)))))]
               (conj '() (fb n)))) 6)
;---
(def fib (map first (iterate (fn [[a b]] [b (+ a b)]) [1 1])))
(take 3 fib)
(defn fibo [n] (def fib (map first (iterate (fn [[a b]] [b (+ a b)]) [1 1]))) (take n fib))
;--
doesnt work 
((fn [s] (tree-seq sequential? identity s)) '((1 2) 3 [4 [5 6]]))
((fn [s] (tree-seq sequential? #(if-not (sequential? %) (identity %)) s)) '((1 2) 3 [4 [5 6]]))

;28
(fn [x] (filter (complement sequential?) (rest (tree-seq sequential? seq x))))
(= ((fn [x] (filter (complement sequential?) (rest (tree-seq sequential? seq x)))) '((1 2) 3 [4 [5 6]])) '(1 2 3 4 5 6))
(= (__ ["a" ["b"] "c"]) '("a" "b" "c"))
(= (__ '((((:a))))) '(:a))
(map #(if-not (sequential? %) %) '((1 2) 3 [4 [5 6]]))
(filter (complement sequential?) '((1 2) 3 [4 [5 6]]))
(#(rest (tree-seq sequential? seq %)) '((1 2) 3 [4 [5 6]]))
(sequential? [1 [2] 3])
(tree-seq sequential? seq '((1 2) 3 [4 [5 6]]))
(tree-seq sequential? identity '((1 2) 3 [4 [5 6]]))

;29
(fn [s]   (re-seq  #"[A-Z]" s))
(fn [s]   (clojure.string/join (re-seq  #"[A-Z]" s)));works!
(fn [s] (str (filter #(if (= % (clojure.string/upper-case %)) %) (seq s))))
(= ((fn [s] (str (filter #(if (= (str %) (clojure.string/upper-case (str %)))  %) (seq s)))) "HeLlO, WoRlD!") "HLOWRD")
(empty? ((fn [s]   (clojure.string/join (re-seq  #"[A-Z]" s))) "nothing"))
(= ((fn [s]   (re-seq  #"[A-Z]" s)) "$#A(*&987Zf") "AZ")
((fn [s] (str (filter #(if (= % (clojure.string/upper-case %))  %) s))) "HeLlO, WoRlD!")
(seq "ii")
(clojure.string/lower-case ",.!@#$%^&*()")
(cmap j)
((fn [s] (str (filter #(re-seq  #"[A-Z]" (str %)) (seq s)))) "dfOgh")
((fn [s]   (re-seq  #"[A-Z]" s)) "dfOgh")
#(Character/isUpperCase %)
(clojure.string/join ((fn [s]   (re-seq  #"[A-Z]" s)) "$#A(*&987Zf"))
(str (seq \O));=>"(\\O)" ???????????

;30
(fn f [s] (letfn
           [ff [el coll] (if not-empty coll
                             (case
                              (not= el (first coll)) el))] (ff (first s) s)))
(rest "Leroy")
(empty)
(condp);???
(not-empty nil)
(boolean 1)
(distinct  [1 1 2 3 3 2 2 3])
(fnext  [1  2 3 3 2 2 3] )
(second [1  2 3 3 2 2 3])

({:keys [b c d] :or {d 10 b 20 c 30}})

(re-seq #"[A-Z]+" "bA1B3Ce ")

;--38.
;; Write a function which takes a variable number of parameters and returns the maximum value.
;;     (= (__ 1 8 3 4) 8)
;;     (= (__ 30 20) 30)
;;     (= (__ 45 67 11) 67)
;; Special Restrictions : max,max-key
(fn [a & b] partial if )
(defn f [a & b] (apply +))
(f 9 8 7)
((fn [a & b] +) 9 8 7 6)

(= (__ 1 8 3 4) 8)
(= (__ 30 20) 30)
(= (__ 45 67 11) 67)
;working variant for a collection
((fn [& s] (let [cs (list s)] (letfn [(r [m coll]
                                         (println (str "m22" m ", coll22:" coll))
                                        (if (empty? coll) 
                     m;(println (str "m22" m "coll22:" coll) )
                     (r (if (> (first coll) m)(first coll) m) (rest coll)))(println (str "m: " m " coll:" coll)))] 
                                 (r (first s) (rest s)))))1 8 2 4 )
m221, coll22:(8 2 4)
m228, coll22:(2 4)
m228, coll22:(4)
m228, coll22:()
m: 8 coll:()
m: 8 coll:(4)
m: 8 coll:(2 4)
m: 1 coll:(8 2 4)
;!!!!!!!!
;unlimited output, broken calva
((fn [& s] (let [cs (list s)] (letfn [(r [m coll]
                                        (println (str "m22" m ", coll22:" coll))
                                        (while (not (empty? coll))                                         
                                          (r (if (> (first coll) m) (first coll) m) (rest coll))) m (println (str "m: " m " coll:" coll)))]
                                (r (first s) (rest s))))) 1 8 2 4)

(#(if % 5 ((print "6") 7)) nil)
(type 8)
((fn [& cs] (let [s (list cs)] (println cs))) 1 8 2 4)
(if (not-empty? [9 9 0]) 8 0)
(list 9 8 7 6)
(empty? ())

;works!
(fn [& s]  (letfn [(r [m coll]  (if (empty? coll)
                                  m
                                  (r (if (> (first coll) m) (first coll) m) (rest coll))))]
             (r (first s) (rest s))))

;--39.
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
((fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                      (rec (conj res (first x) (first y)) (rest x) (rest y))
                                      res))]
               (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy)))) [1 2 3] [:a :b :c])
(= ((fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                         (rec (conj res (first x) (first y)) (rest x) (rest y))
                                         res))]
                  (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy)))) [1 2 3] [:a :b :c]) '(1 :a 2 :b 3 :c))
(= ((fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                         (rec (conj res (first x) (first y)) (rest x) (rest y))
                                         res))]
                  (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy)))) [1 2] [3 4 5 6]) '(1 3 2 4))
(= ((fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                         (rec (conj res (first x) (first y)) (rest x) (rest y))
                                         res))]
                  (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy)))) [1 2 3 4] [5]) [1 5])
(= ((fn [xx yy] (letfn [(rec [res x y] (if (and (not (empty? x)) (not (empty? y)))
                                         (rec (conj res (first x) (first y)) (rest x) (rest y))
                                         res))]
                  (rec (conj (empty xx) (first xx) (first yy)) (rest xx) (rest yy)))) [30 20] [25 15]) [30 25 20 15])
(conj 9 [8 0])
(nth [8 7 9])
(nthrest)
(conj (empty [1 2 3]) (first [:a :b :c]) (first [1 2 3]))
(empty [9 6 5])
(zipmap)
;???? why???? (1 :a 2 :b 3 :c) shouldn't be equal to [1 :a 2 :b 3 :c]

;from clojure.core
(defn interleave
  "Returns a lazy seq of the first item in each coll, then the second etc."
  {:added "1.0"
   :static true}
  ([] ())
  ([c1] (lazy-seq c1))
  ([c1 c2]
   (lazy-seq
    (let [s1 (seq c1) s2 (seq c2)]
      (when (and s1 s2)
        (cons (first s1) (cons (first s2)
                               (interleave (rest s1) (rest s2))))))))
  ([c1 c2 & colls]
   (lazy-seq
    (let [ss (map seq (conj colls c2 c1))]
      (when (every? identity ss)
        (concat (map first ss) (apply interleave (map rest ss))))))))

(replace)
(cons)

;??????
 (-> "a b c d"
     .toUpperCase
     (.replace "A" "X")
     (.split " ")
     first); => "X"
;=>(first (.split (.replace (.toUpperCase "a b c d") "A" "X") " "))
(->> "a b c d"
     .toUpperCase
     (.replace "A" "X")
     (.split " ")
     first); => " "
(use 'clojure.walk)
(macroexpand-all '(->> "a b c d"
                       .toUpperCase
                       (.replace "A" "X")
                       (.split " ")
                       first))
;=>(first (. " " split (. "A" replace "X" (. "a b c d" toUpperCase))))
(. "a b c d" toUpperCase)
(. "a b c d" toUpperCase)
(. replace "A B C D" "A" "X")
(. "A B C D" ("A" replace "X"))
(. "a b c d" toUpperCae)
;; user=> (def c 5)
;; user=> (->> c (+ 3) (/ 2) (- 1))
;; 3/4
;; ;; and if you are curious why
;; user=> (use 'clojure.walk)
;; user=> (macroexpand-all '(->> c (+ 3) (/ 2) (- 1)))
;; (- 1 (/ 2 (+ 3 c)))

;--40.
;; Write a function which separates the items of a sequence by an arbitrary value.
;;     (= (__ 0 [1 2 3]) [1 0 2 0 3])
;;     (= (apply str (__ ", " ["one" "two" "three"])) "one, two, three")
;;     (= (__ :z [:a :b :c :d]) [:a :z :b :z :c :z :d])
;; Special Restrictions : interpose
(drop-last)
(map  #(conj (empty [1 2 3]) % 0) [1 2 3])
;works, but long
(fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (reduce concat (map #(conj (empty sx) % sep) sx))))
;works
(fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (mapcat #(conj (empty sx) % sep) sx)))
(= ((fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (mapcat #(conj (empty sx) % sep) sx))) 0 [1 2 3]) [1 0 2 0 3])
(= (apply str ((fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (mapcat #(conj (empty sx) % sep) sx))) ", " ["one" "two" "three"])) "one, two, three")
(= ((fn [sep sx] (drop-last (if (counted? sep) (count sep) 1) (mapcat #(conj (empty sx) % sep) sx))) :z [:a :b :c :d]) [:a :z :b :z :c :z :d])

(any?)
'[9 8]

;--41.
;; Write a function which drops every Nth item from a sequence.
;; (= (__ [1 2 3 4 5 6 7 8] 3) [1 2 4 5 7 8])
;; (= (__ [:a :b :c :d :e :f] 2) [:a :c :e])
;; (= (__ [1 2 3 4 5 6] 4) [1 2 3 5 6])
;doesnt work
(fn [s n]
  (def rr (cycle (reverse (range n))))
  (letfn [(foo [r res sx] (if sx
                            (if (not= 0 (first r)) ((rest r) (conj res (first sx)) (rest sx)))
                            res))]
    foo ((first rr) (first s) (next s))))
;doesnt work 
(fn [s n]
  (letfn [(foo [r res sx] (if (not-empty sx)
                            (if (not= 0 (first r))
                              (foo (rest r) (conj res (first sx)) (rest sx)))
                            res))]
    (foo (cycle (reverse (range n))) (empty s) s)))
(take 8 (cycle (reverse (range 3))))
(rever)
(def rr (cycle (reverse (range 3))))
(type rr)
(def rr (cycle (reverse (range 15))))
(take 20 (rest rr))
(seq (rest [7]))
(boolean (not-empty nil))
(conj [] 0)
(= (__ [1 2 3 4 5 6 7 8] 3) [1 2 4 5 7 8])
(= (__ [:a :b :c :d :e :f] 2) [:a :c :e])
(= (__ [1 2 3 4 5 6] 4) [1 2 3 5 6])
(take 8 (cycle (reverse (range 3))))
(rever)
(def rr (cycle (reverse (range 3))))
(type rr)
;works!
((fn [s n]
   (letfn [(foo [r res sx] (if (not-empty sx)
                             (if (not= 0 (first r))
                               (do
                                 (println (str "rest r: " (take 5 (rest r)) ", res: " res ", sx " sx))
                                 (foo (rest r) (conj res (first sx)) (rest sx)))
                               (foo (rest r) res  (rest sx)))
                             res))]
     (foo (cycle (reverse (range n))) (empty s) s))) [1 2 3 4 5 6 7 8] 3)


;--43.
;; Write a function which reverses the interleave process into x number of subsequences.
;; (= (__ [1 2 3 4 5 6] 2) '((1 3 5) (2 4 6)))
;; (= (__ (range 9) 3) '((0 3 6) (1 4 7) (2 5 8)))
;; (= (__ (range 10) 5) '((0 5) (1 6) (2 7) (3 8) (4 9)))
(fn [s n]
  (let [r (cycle (reverse (range n)))
        start (into [] (repeat n '()))]
    (letfn [(foo [rr res sx] (if (not-empty sx)
                               (foo (rest rr) (update res  (first rr) #(cons  (first sx) %)) (rest sx))
                               res))]
      (reverse (map reverse (seq (foo r start s)))))))
(into [] (repeat 5 '()))
(cycle (reverse (range n)))
(def a '('(45 4) '(5 66)))
(a 0)
(update ['(45 4) '(5 66)] 1   '(5 66 11))
((fn [s n] (partition (/ (count s) n)  s)) [1 2 3 4 5 6] 2)

;--44.
;; Write a function which can rotate a sequence in either direction.
;; (= (__ 2 [1 2 3 4 5]) '(3 4 5 1 2))
;; (= (__ -2 [1 2 3 4 5]) '(4 5 1 2 3))
;; (= (__ 6 [1 2 3 4 5]) '(2 3 4 5 1))
;; (= (__ 1 '(:a :b :c)) '(:b :c :a))
;; (= (__ -4 '(:a :b :c)) '(:c :a :b))
(fn [n s] (letfn [(foo [c m] (cond
                               (= m 0) c
                               (> m 0) (foo (reverse (conj  (reverse (rest c)) (first c))) (dec m))
                               (< m 0) (foo (cons (last c) (butlast c)) (inc m))))] (foo s n)))
(cons (last [1 2 3 4 5]) (butlast [1 2 3 4 5]))
(conj  (rest '(1 2 3 4 5)) (first '(1 2 3 4 5)))
(cons (peek '(1 2 3 4 5)) (pop '(1 2 3 4 5)))
(peek '(1 2 3 4 5))
(pop '(1 2 3 4 5))

;--46.----------------------------
;; Write a higher-order function which flips the order of the arguments of an input function.
;; (= 3 ((__ nth) 2 [1 2 3 4 5]))
;; (= true ((__ >) 7 8))
;; (= 4 ((__ quot) 2 8))
;; (= [1 2 3] ((__ take) [1 2 3 4 5] 3))
;map from clojure.core
(defn map
  "Returns a lazy sequence consisting of the result of applying f to
  the set of first items of each coll, followed by applying f to the
  set of second items in each coll, until any one of the colls is
  exhausted.  Any remaining items in other colls are ignored. Function
  f should accept number-of-colls arguments. Returns a transducer when
  no collection is provided."
  {:added "1.0"
   :static true}
  ([f]
   (fn [rf]
     (fn
       ([] (rf))
       ([result] (rf result))
       ([result input]
        (rf result (f input)))
       ([result input & inputs]
        (rf result (apply f input inputs))))))
  ([f coll]
   (lazy-seq
    (when-let [s (seq coll)]
      (if (chunked-seq? s)
        (let [c (chunk-first s)
              size (int (count c))
              b (chunk-buffer size)]
          (dotimes [i size]
            (chunk-append b (f (.nth c i))))
          (chunk-cons (chunk b) (map f (chunk-rest s))))
        (cons (f (first s)) (map f (rest s)))))))
  ([f c1 c2]
   (lazy-seq
    (let [s1 (seq c1) s2 (seq c2)]
      (when (and s1 s2)
        (cons (f (first s1) (first s2))
              (map f (rest s1) (rest s2)))))))
  ([f c1 c2 c3]
   (lazy-seq
    (let [s1 (seq c1) s2 (seq c2) s3 (seq c3)]
      (when (and  s1 s2 s3)
        (cons (f (first s1) (first s2) (first s3))
              (map f (rest s1) (rest s2) (rest s3)))))))
  ([f c1 c2 c3 & colls]
   (let [step (fn step [cs]
                (lazy-seq
                 (let [ss (map seq cs)]
                   (when (every? identity ss)
                     (cons (map first ss) (step (map rest ss)))))))]
     (map #(apply f %) (step (conj colls c3 c2 c1))))))

([:a :c] [:b :d])
(apply)
(defmacro make-fn [m]
  `(fn [& args#]
     (eval
      (cons '~m args#))))
(apply (make-fn and) '(true true false true))
(defmacro make-fn [m]
  `(fn [a b]
     (eval
      (cons '~m b a))))
(apply (make-fn cons) '(4 [0 1 2]))

;--50.
;; Write a function which takes a sequence consisting of items with different types and splits them up 
;; into a set of homogeneous sub-sequences. The internal order of each sub-sequence should be maintained, 
;; but the sub-sequences themselves can be returned in any order (this is why 'set' is used in the test cases) .
;; (= (set (__ [1 :a 2 :b 3 :c])) #{[1 2 3] [:a :b :c]})
;; (= (set (__ [:a "foo"  "bar" :b])) #{[:a :b] ["foo" "bar"]})
;; (= (set (__ [[1 2] :a [3 4] 5 6 :b])) #{[[1 2] [3 4]] [:a :b] [5 6]})
(fn [s] (mapcat #(hash-map (type %) %) s))
((fn [s] (mapcat #(hash-map (type %) %) s)) [1 :a 2 :b 3 :c])
;=>([java.lang.Long 1] [clojure.lang.Keyword :a] [java.lang.Long 2] [clojure.lang.Keyword :b] [java.lang.Long 3] [clojure.lang.Keyword :c])
(fn [s] (vals (group-by type s)))
((fn [s] (vals (group-by type s))) [1 :a 2 :b 3 :c])
;=>{java.lang.Long [1 2 3], clojure.lang.Keyword [:a :b :c]}
((fn [s] (map type s)) [1 :a 2 :b 3 :c])
(= (set ((fn [s] (vals (group-by type s))) [1 :a 2 :b 3 :c])) #{[1 2 3] [:a :b :c]})
(= (set ((fn [s] (vals (group-by type s))) [:a "foo"  "bar" :b])) #{[:a :b] ["foo" "bar"]})
(= (set ((fn [s] (vals (group-by type s))) [[1 2] :a [3 4] 5 6 :b])) #{[[1 2] [3 4]] [:a :b] [5 6]})

;--53. 
;; Given a vector of integers, find the longest consecutive sub-sequence of increasing numbers. 
;; If two sub-sequences have the same length, use the one that occurs first. 
;; An increasing sub-sequence must have a length of 2 or greater to qualify.
;; (= (__ [1 0 1 2 3 0 4 5]) [0 1 2 3])
;; (= (__ [5 6 1 3 2 7]) [5 6])
;; (= (__ [2 3 3 4 5]) [3 4 5])
;; (= (__ [7 6 5 4]) [])
(partition)
(partition-all)
(partition-by)
(reduce-kv)
((fn [s]
   (letfn [(cf [res ss]
             (clojure.pprint/pprint res)
             (clojure.pprint/pprint ss)
             (if (not-empty ss)
               (if (> (first ss) (last (last res)))
                 (cf (update  res (- (count res) 1) #(conj % (first ss))) (rest ss))
                 (cf (conj res (vector (first ss))) (rest ss)))
               res))]
     (let [cs (cf [[(first s)]] (rest s))]
       (let [result (first (val (apply max-key key (group-by count cs))))]
         (if (> (count result) 1) result []))))) [5 6 1 3 2 7])
(vector (vector (first [1 2 3])))
(last (last [[9]]))
(vector (vector))
(#(inc %) 9)
(>9 nil)

(-> (map inc [1 2 3])
    (vec))
;=> [2 3 4]
(for [[k v] {:a 1 :b 2 :c 3}]
  [k (inc v)])

;--55.
;Partition a Sequence
;; Write a function which returns a map containing the number of occurences of each distinct item in a sequence.
;;     (= (__ [1 1 2 3 2 1 1]) {1 4, 2 2, 3 1})
;;     (= (__ [:b :a :b :a :b]) {:a 2, :b 3})
;;     (= (__ '([1 2] [1 3] [1 3])) {[1 2] 1, [1 3] 2})
;; Special Restrictions : frequencies
(fn [s] ())
(group-by identity [1 1 2 3 2 1 1]); => {1 [1 1 1 1], 2 [2 2], 3 [3]}
(map second (group-by identity [1 1 2 3 2 1 1])); => ([1 1 1 1] [2 2] [3])
(map #(-> % (second) (count)) (group-by identity [1 1 2 3 2 1 1])); =>(4 2 1)
(first %) (count (second %))

(let [r (group-by identity [1 1 2 3 2 1 1])] (assoc r (keys r) (vals r)));=>{1 [1 1 1 1], 2 [2 2], 3 [3], (1 2 3) ([1 1 1 1] [2 2] [3])}
(let [r (group-by identity [1 1 2 3 2 1 1])] (map (fn [a b] (vector a (count b))) r))
(println r)
(map #(hash-map (first %) (count (second %)))
     (group-by identity [1 1 2 3 2 1 1])); =>({1 4} {2 2} {3 1})
;works!
(reduce conj (map #(hash-map (first %) (count (second %)))
                  (group-by identity [1 1 2 3 2 1 1])))
(fn [s] (reduce conj (map #(hash-map (first %) (count (second %)))
                          (group-by identity s))))

(map #(assoc  (first %) (count (second %)))
     (group-by identity [1 1 2 3 2 1 1]));=> error
(map (fn [a b] (a (count b)))
     (group-by identity [1 1 2 3 2 1 1]));=> error
(mapcat #(hash-map (first %) (count (second %)))
        (group-by identity [1 1 2 3 2 1 1])); =>([1 4] [2 2] [3 1])
(contains?)
(frequencies)
(= (__ [1 1 2 3 2 1 1]) {1 4, 2 2, 3 1})
(= (__ [:b :a :b :a :b]) {:a 2, :b 3})
(= (__ '([1 2] [1 3] [1 3])) {[1 2] 1, [1 3] 2})
(hash-map)
([1 4])
()

;--56.
;Find Distinct Items
;; Write a function which removes the duplicates from a sequence. Order of the items must be maintained.
;;     (= (__ [1 2 1 3 1 2 4]) [1 2 3 4])
;;     (= (__ [:a :a :b :b :c :c]) [:a :b :c])
;;     (= (__ '([2 4] [1 2] [1 3] [1 3])) '([2 4] [1 2] [1 3]))
;;     (= (__ (range 50)) (range 50))
;; Special Restrictions : distinct
(fn d [v] (into (empty v) (apply sorted-set (into #{} v))))
((fn d [v] (into [] (apply sorted-set (into #{} v)))) [1 2 1 3 1 2 4])
((fn d [v] (into (empty v)  (into #{} v))) '([2 4] [1 2] [1 3] [1 3]))
(iterate (fn [v res] (let [res ()] (map #(if (some (= %)) identity) v))) '([2 4] [1 2] [1 3] [1 3]))
((fn [v res] (let [res ()] (map #(if (some (= %)) identity) v))) '([2 4] [1 2] [1 3] [1 3]))
(fn d [v] (sort (into (empty v) (apply sorted-set (into #{} v)))))
(fn d [v] (map #(if (some (= %)) identity)))
([9 0])
((fn [s] (keys (group-by identity s))) (range 50))
(keys (group-by identity '([2 4] [1 2] [1 3] [1 3])))
;works!
(fn [s] (letfn [(foo [ss res] (if (not-empty ss)
                                (if (some #(= (first ss) %) res)
                                  (foo (rest ss) res)
                                  (foo (rest ss) (conj res (first ss))))
                                (into (empty s) (if  (vector? s)
                                                  (reverse res)
                                                  res))))] (foo s ())))


;--58.
;; Write a function which allows you to create function compositions. 
;; The parameter list should take a variable number of functions, and create a function applies them from right-to-left.
;;     (= [3 2 1] ((__ rest reverse) [1 2 3 4]))
;;     (= 5 ((__ (partial + 3) second) [1 2 3 4]))
;;     (= true ((__ zero? #(mod % 8) +) 3 5 7 9))
;;     (= "HELLO" ((__ #(.toUpperCase %) #(apply str %) take) 5 "hello world"))
;; Special Restrictions : comp
; from Clojure.core
(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (fn [& args]
    (reduce #(%2 %1)
            (apply (first ff) args)
            (rest ff))))

(fn [s] (fn [arg]
          (letfn [(foo [ss](if (not-empty ss) 
                           ((last ss) (foo (butlast ss)))
                           arg ))]
          (foo s))))

((
  (fn [& s] (fn [arg]
            (letfn [(foo [ss] (if (not-empty ss)
                                ((first ss) (foo (rest ss)))
                                arg))]
              (into (empty arg)(foo s))))) rest reverse) [1 2 3 4])
(last '(rest reverse))

(= [3 2 1] (((fn [& s] (fn [arg]
                         (letfn [(foo [ss] (if (not-empty ss)
                                             ((first ss) (foo (rest ss)))
                                             arg))]
                           (into (empty arg) (foo s))))) rest reverse) [1 2 3 4]))
;doesnt work
(= 5 (((fn [& s] (fn [arg]
                   (letfn [(foo [ss] (if (not-empty ss)
                                       ((first ss) (foo (rest ss)))
                                       arg))]
                     (into (empty arg) (foo s))))) (partial + 3) second) [1 2 3 4]))
;works
(((fn [& s] (fn [& arg]
              (println (first arg))
              (letfn [(foo [ss]
                        (println ss)
                        (if (not-empty ss)
                          ((first ss)
                           (foo (rest ss)))
                          (first arg)))]
                (if (seq? (first arg))
                  (into (empty (first arg)) (foo s))
                  (foo  s)))))
  (partial + 3) second) [1 2 3 4])
;works
(= [3 2 1] (((fn [& s] (fn [& arg]
                         (println (first arg))
                         (letfn [(foo [ss]
                                   (println ss)
                                   (if (not-empty ss)
                                     ((first ss)
                                      (foo (rest ss)))
                                     (first arg)))]
                           (if (seq? (first arg))
                             (into (empty (first arg)) (foo s))
                             (foo  s))))) rest reverse) [1 2 3 4]))
;doesnt work 
(= "HELLO" (((fn [& s] (fn [& arg]
                         (let [a 
                               (if (>(count arg)1)
                                 arg
                                 (first arg))
                              ;;  (if (some seqable? arg)
                              ;;      (first arg)
                              ;;      arg)
                               ]
                           (println a)
                           (letfn [(foo [ss]
                                     (println ss)
                                     (if (not-empty ss)
                                       ((first ss)
                                        (foo (rest ss)))
                                       a))]
                             (let  [result (foo s)]
                               (println "result" result)
                               result
                              ;;  (if (seqable? arg)
                              ;;    (into (empty (first arg)) result)
                              ;;    result)
                               )))))
             #(.toUpperCase %) 
             #(apply str %) 
             take) 5 "hello world"))

 (5 hello world)
(#function[clojure.core/take] #function[user/eval8980/fn--8989] #function[user/eval8980/fn--8987])
(#function[user/eval8980/fn--8989] #function[user/eval8980/fn--8987])
(#function[user/eval8980/fn--8987])
()
; Execution error (IllegalArgumentException) at user/eval8980$fn (REPL:216).
; No matching field found: toUpperCase for class clojure.lang.ArraySeq
 (equal? (user/eval8980/fn--8989 6) clojure.core/take)
(clojure.core/take 3 [5 6 7 8]) ; =>(5 6 7)
 (take 5 "hello world")

 (5 hello world)
(#function[user/eval9012/fn--9019] #function[user/eval9012/fn--9021] #function[clojure.core/take])
(#function[user/eval9012/fn--9021] #function[clojure.core/take])
(#function[clojure.core/take])
()
; Execution error (IllegalArgumentException) at user/eval9012$fn (REPL:217).
; Don't know how to create ISeq from: clojure.core$take$fn__5923

;doesnt work
(((fn [& s] (fn [& arg]
              (let [a (if (some seqable? arg)
                        (do (println "yes") (first arg))
                        (do (println (type arg)) arg))]
                (println a)
                (letfn [(foo [ss]
                          (println ss)
                          (if (not-empty ss)
                            ( (first ss)
                             (foo (rest ss)))
                            (do 
                              (println a)
                            a)))]
                  (let  [result (foo  s)]
                    (println result)
                    (if (seqable? a)
                      (into (empty (first arg)) result)
                      result))))))
  ;; zero? 
  ;; #(mod % 8) 
  +) 
 3 5 7 9)
;; => Execution error (IllegalArgumentException) at user/eval7700$fn$fn$foo (REPL:221).
;;    Don't know how to create ISeq from: java.lang.Long
(apply + '( 3 5 7 9))

(not-empty ())
clojure.lang.ArraySeq
(empty? 9)


;https://stackoverflow.com/questions/25628724/misunderstanding-of-variable-arguments-type
;1
(defn my-comp [& fns]
  (if (empty? fns)
    identity
    (let [[f & fs] fns]
      (fn [& args] (reduce #(%2 %1) (apply f args) fs)))))
(fn  [& fns]
  (if (empty? fns)
    identity
    (let [[f & fs] fns]
      (fn [& args] (reduce #(%2 %1) (apply f args) fs)))))
;2
(defn my-comp [& fncs]
  (fn [& args]
    (reduce #(%2 %1) ; you can omit apply here, as %2 is already function
                           ; and %1 is always one value, as noisesmith noticed
            (apply (first fncs) args)
            (rest fncs))))
(fn [& fncs]
  (fn [& args]
    (reduce #(%2 %1) 
            (apply (first fncs) args)
            (rest fncs))))

(reduce into [][[10 18] [8 18] [10 12] [0 -6] [2 6]]);=>[10 18 8 18 10 12 0 -6 2 6]

(((fn  [& fns]
   (if (empty? fns)
     identity
     (let [[f & fs] fns]
       (fn [& args] (println args)
         (reduce #(%2 %1) (apply f args) fs)))))rest reverse) [1 2 3 4])

    (= [3 2 1] ((fn  [& fns]
                  (if (empty? fns)
                    identity
                    (let [[f & fs] fns]
                      (fn [& args] ;(println "args")
                        (reduce #(%2 %1) (apply f args) fs))))) [1 2 3 4]))
    (= 5 ((__ (partial + 3) second) [1 2 3 4]))
    (= true ((__ zero? #(mod % 8) +) 3 5 7 9))
    (= "HELLO" ((__ #(.toUpperCase %) #(apply str %) take) 5 "hello world"))
(((fn  [& fns]
    (if (empty? fns)
      identity
      (let [[f & fs] fns]
        (fn [& args] (reduce #(%2 %1) (apply f args) fs))))) rest reverse) [1 2 3 4])
(((fn [& fncs]
    (fn [& args]
      (reduce #(%2 %1)
              (apply (first fncs) args)
              (rest fncs)))) rest reverse) [1 2 3 4])
;=>(4 3 2)
(((fn  [& fns]
    (if (empty? fns)
      identity
      (let [[f & fs] (reverse  fns)]
        (fn [& args] (println (empty (first args))) (reduce #(%2 %1) (apply f args) fs)))))rest reverse) [1 2 3 4])
    
(((fn  [& fns]
    (if (empty? fns)
      identity
      (let [[f & fs] (reverse  fns)]
        (fn [& args] (println (empty (first args))) (reduce #( %2 %1) (first args) (reverse fns)))))) rest reverse) [1 2 3 4])
;=>(3 2 1)

    (into (empty "i") [9])
    (empty "i")

;https://stackoverflow.com/questions/2020570/common-programming-mistakes-for-clojure-developers-to-avoid
(doseq)
(doall)
(dorun)

user=> (defn foo [] (doseq [x [:foo :bar]] (println x)) nil)
#'user/foo
user=> (foo)
:foo
:bar
nil
user=> (defn foo [] (dorun (map println [:foo :bar])) nil)
#'user/foo
user=> (foo)
:foo
:bar
nil
    

(defn comp
  "Takes a set of functions and returns a fn that is the composition
  of those fns.  The returned fn takes a variable number of args,
  applies the rightmost of fns to the args, the next
  fn (right-to-left) to the result, etc."
  {:added "1.0"
   :static true}
  ([] identity)
  ([f] f)
  ([f g]
   (fn
     ([] (f (g)))
     ([x] (f (g x)))
     ([x y] (f (g x y)))
     ([x y z] (f (g x y z)))
     ([x y z & args] (f (apply g x y z args)))))
  ([f g & fs]
   (reduce1 comp (list* f g fs))))
;; reduce is defined again later after InternalReduce loads
(defn ^:private ^:static
  reduce1
  ([f coll]
   (let [s (seq coll)]
     (if s
       (reduce1 f (first s) (next s))
       (f))))
  ([f val coll]
   (let [s (seq coll)]
     (if s
       (if (chunked-seq? s)
         (recur f
                (.reduce (chunk-first s) f val)
                (chunk-next s))
         (recur f (f val (first s)) (next s)))
       val))))

;
(fn [& ff]
  (letfn [(reduce1
            ([f coll]
             (let [s (seq coll)]
               (if s
                 (reduce1 f (first s) (next s))
                 (f))))
            ([f val coll]
             (let [s (seq coll)]
               (if s
                 (if (chunked-seq? s)
                   (recur f
                          (.reduce (chunk-first s) f val)
                          (chunk-next s))
                   (recur f (f val (first s)) (next s)))
                 val))))]
    (letfn  [(compose
               ([] identity)
               ([f] f)
               ([f g]
                (fn
                  ([] (f (g)))
                  ([x] (f (g x)))
                  ([x y] (f (g x y)))
                  ([x y z] (f (g x y z)))
                  ([x y z & args] (f (apply g x y z args)))))
               ([f g & fs]
                (reduce1 compose (list* f g fs))))] (compose ff))))

(= [3 2 1] (((fn [f g & fs]
                   ;(def [(f g fs) ff])
               (letfn [(reduce1
                         ([f coll]
                          (let [s (seq coll)]
                            (if s
                              (reduce1 f (first s) (next s))
                              (f))))
                         ([f val coll]
                          (let [s (seq coll)]
                            (if s
                              (if (chunked-seq? s)
                                (recur f
                                       (.reduce (chunk-first s) f val)
                                       (chunk-next s))
                                (recur f (f val (first s)) (next s)))
                              val))))]
                 (letfn  [(compose
                            ([] identity)
                            ([f] f)
                            ([f g]
                             (fn
                               ([] (f (g)))
                               ([x] (f (g x)))
                               ([x y] (f (g x y)))
                               ([x y z] (f (g x y z)))
                               ([x y z & args] (f (apply g x y z args)))))
                            ([f g & fs]
                             (reduce1 compose (list* f g fs))))]
                   (compose)))) rest reverse) [1 2 3 4]))
; Execution error (ClassCastException) at user/eval9135 (REPL:209).
; clojure.lang.Cons cannot be cast to clojure.lang.IFn

(((fn [f g & fs]
    (letfn [(reduce1
              ([f coll]
               (let [s (seq coll)]
                 (if s
                   (reduce1 f (first s) (next s))
                   (f))))
              ([f val coll]
               (let [s (seq coll)]
                 (if s
                   (if (chunked-seq? s)
                     (recur f
                            (.reduce (chunk-first s) f val)
                            (chunk-next s))
                     (recur f (f val (first s)) (next s)))
                   val))))]
      (letfn  [(compose
                 ([] identity)
                 ([f] f)
                 ([f g]
                  (fn
                    ([] (f (g)))
                    ([x] (f (g x)))
                    ([x y] (f (g x y)))
                    ([x y z] (f (g x y z)))
                    ([x y z & args] (f (apply g x y z args)))))
                 ([f g & fs]
                  (reduce1 compose (list* f g fs))))]
        (compose)))) rest reverse) [1 2 3 4])




(fn [a]
  (fn [args]
    (->>
     args
     (map eval (reverse a)))))
(eval (+ 6 7))

(= 5 (((fn [f g & fs]
         (def ff (list* f g fs))
         (fn [args]
           (->>
            args
            (map list (reverse ff)))))
       (partial + 3) second) [1 2 3 4]))
(((fn [f g & fs]
    (def ff (reverse (list* f g fs)))
    (fn [args]
      (println ff)
      (println args)
      (-> ;>
       args
       ;(map #((%)) ff)
       (take-while fn? ff)
       ;(map list (reverse ff))
       )))
  (partial + 3) second) [1 2 3 4])

(= true ((__ zero? #(mod % 8) +) 3 5 7 9))
(= "HELLO" ((__ #(.toUpperCase %) #(apply str %) take) 5 "hello world"))

(lazy-seq [1 2 3])
(take-while fn? [1 2 3])




(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (fn [args]
    (reduce #(%2 %1)
            (apply (first ff) args)
            (rest ff))))



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOORKS!!!!!!!!!!!!!!!!!!!
(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (fn [& args]
    (reduce #(%2 %1)
            (apply (first ff) args)
            (rest ff))))

(((fn [f g & fs]
    (def ff (reverse (list* f g fs)))
    (fn [& args]
      (reduce #(%2 %1)
              (apply (first ff) args)
              (rest ff))))
  (partial + 3) second) [1 2 3 4])



(= [3 2 1] (((fn [f g & fs]
               (def ff (reverse (list* f g fs)))
               (fn [& args]
                 (reduce #(%2 %1)
                         (apply (first ff) args)
                         (rest ff)))) rest reverse) [1 2 3 4]))
(= 5 (((fn [f g & fs]
         (def ff (reverse (list* f g fs)))
         (fn [& args]
           (reduce #(%2 %1)
                   (apply (first ff) args)
                   (rest ff)))) (partial + 3) second) [1 2 3 4]))
(= true (((fn [f g & fs]
            (def ff (reverse (list* f g fs)))
            (fn [& args]
              (reduce #(%2 %1)
                      (apply (first ff) args)
                      (rest ff)))) zero? #(mod % 8) +) 3 5 7 9))
(= "HELLO" (((fn [f g & fs]
               (def ff (reverse (list* f g fs)))
               (fn [& args]
                 (reduce #(%2 %1)
                         (apply (first ff) args)
                         (rest ff)))) #(.toUpperCase %) #(apply str %) take) 5 "hello world"))

;--59.
;Juxtaposition
;; Take a set of functions and return a new function that takes a variable number of arguments 
;; and returns a sequence containing the result of applying each function left-to-right to the argument list.
;;     (= [21 6 1] ((__ + max min) 2 3 5 1 6 4))
;;     (= ["HELLO" 5] ((__ #(.toUpperCase %) count) "hello"))
;;     (= [2 6 4] ((__ :a :c :b) {:a 2, :b 4, :c 6, :d 8 :e 10}))
;; Special Restrictions : juxt
(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (let [result []]
    (fn [& args]
      (reduce
       (fn [a b]
         (let [r (b a)]
           (do
             (println "r: " r)
             (def k (conj result r))
             (println "k: " k)
             (b a))))
       (apply (first ff) args)
       (rest ff))
      result)))

(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (let [result []]
    (fn [& args]
      (conj result
            (map
             #(% args) ff)))))

;works!!!!
(fn [f g & fs]
  (def ff (list* f g fs))
  (let [result []]
    (fn [& args]
      (into []
            (map
             #(apply % args) ff)))))

(= [21 6 1] (((fn [f g & fs]
                (def ff (list* f g fs))
                (let [result []]
                  (fn [& args]
                    (into []
                          (map
                           #(apply % args) ff))))) + max min) 2 3 5 1 6 4))
(= ["HELLO" 5] (((fn [f g & fs]
                   (def ff (list* f g fs))
                   (let [result []]
                     (fn [& args]
                       (into []
                             (map
                              #(apply % args) ff))))) #(.toUpperCase %) count) "hello"))
(= [2 6 4] (((fn [f g & fs]
               (def ff (list* f g fs))
               (let [result []]
                 (fn [& args]
                   (into []
                         (map
                          #(apply % args) ff))))) :a :c :b) {:a 2, :b 4, :c 6, :d 8 :e 10}))
;later I accidentally met https://stackoverflow.com/questions/61028339/calling-a-function-within-conj-in-clojure/61028500#61028500
;; (defn  foo [& fns]
;;   #(vector (apply (first fns) %&)))


;--60.
;; Write a function which behaves like reduce, but returns each intermediate value of the reduction. 
;; Your function must accept either two or three arguments, and the return sequence must be lazy.
;;     (= (take 5 (__ + (range))) [0 1 3 6 10])
;;     (= (__ conj [1] [2 3 4]) [[1] [1 2] [1 2 3] [1 2 3 4]])
;;     (= (last (__ * 2 [3 4 5])) (reduce * 2 [3 4 5]) 120)
;; Special Restrictions : reductions
(fn [f g & fs]
  (def ff (reverse (list* f g fs)))
  (let [result []]
    (fn [& args]
      (reduce
       (fn [a b]
         (let [r (b a)]
           (do
             (println "r: " r)
             (def k (conj result r))
             (println "k: " k)
             (b a))))
       (apply (first ff) args)
       (rest ff))
      result)))

(take 5 ((fn [f g & fs]
           (def ff (reverse (list* f g fs)))
           (let [result []]
             (fn [& args]
               (reduce
                (fn [a b]
                  (let [r (b a)]
                    (do
                      (println "r: " r)
                      (def k (conj result r))
                      (println "k: " k)
                      ;(apply b a)
                      )))
                (apply (first ff) args)
                (rest ff))
               result))) + (range)))

(fn
  ([fa]
   (fn [args]
     (reduce fa args)))
  ([fa fb]
   (fn [args]
     (reduce fa fb args))))

(while)

((fn
   ([fa args]
    (let  [result []]
      (letfn [(func! [a b]
                (do
                  (conj result (fa a b))
                  (println (conj result (fa a b)))
                  (fa a b)))]
        (do             ;(println args)     (println (reduce fa args))
          (reduce func! args)
          result)     ;(fa (first args))
        )))
   ([fa fb args] (do (reduce fa fb args)))) + [1 2 3])

(defn pp [fa & fb] (println fa fb))
(pp 9 8 7)

((fn [f args]
   (letfn [(foo [args] if (second args)
             (foo)
             println (first args))])) + [1 2 3])


((fn [fa args]
   (for [x args
         y (rest args)
         :let [z (fa x y)]
       ;:let [i 9]
         :while (not= x y)]
     z
   ;(lazy-seq (cons (first args) z) )
  ;;  (do ;(println z)
  ;; (def res (conj  z (first args)))
  ;;    res)
     ))+ [1 2 3])
;=>(3 4 4 5 5 6)

((fn [fa args]
   (letfn  [(foo [f a b]
              (let [r (f a (first b))]
                (while (not-empty b) (println r)
                       (foo f r (rest b))))
         ;(conj  z (first args))
              )]
        ;;  (iterate
        ;;   (foo fa (first args) (rest args))
        ;;   args)         
     (foo fa (first args) (rest args))))
 + [1 2 3])

(boolean nil)
??????????????????????????
(boolean (rest [])) ;=> true
(boolean (next [])) ;=> false
(boolean (first [])) ;=> false
(boolean (second [])) ;=> false
(boolean (rest '())) ;=> true
(boolean (next '())) ;=> false
(boolean (first '())) ;=> false
(boolean (second '())) ;=> false
(not-empty (second []))


((fn
   [f & arg]
   (def args (first arg))
   ;(println "11111 "(first args))
   (letfn [(foo [result a s]
             (if (not-empty s)
               (do (println "result" result)
                              ;;  (println "rest: " (rest s))
                              ;;  (println "(f a (first s))"(f a (first s)))
                              ;;  (println "conj"(conj result (f a (first s))))
                   (foo (conj result (f a (first s)))
                        (first s) (rest s)))
               result
                          ;;  (do (println "result1" result)
                          ;;      (println "rest: " (rest s))
                          ;;      (println "result1" result)
                          ;;      (lazy-seq result))
               ))]
      ;(println (rest args))
     (foo [(first args)] (first args) (rest args)))   ;[(first args)]
   ;([fa fb args] (do (println args) (reduce fa fb args)))
   )+ (range 5))
lazy-seq
(range 5)

((fn
   [f & arg]
   (def args (first arg))
   ;(println "11111 "(first args))
   (letfn [(foo [result a s]
             (if (not-empty s)
               (do (println "a" a ", s" s ", result" result)
                   (foo (conj result  (apply f a (first s) (butlast result)))
                        (first s) (rest s)))
               result))]
     (foo [(first args)] (first args) (rest args)))   ;[(first args)]
   ;([fa fb args] (do (println args) (reduce fa fb args)))
   )+ (range 5))
;=>[0 1 3 6 11]
(range 5);=>(0 1 2 3 4)

((fn
   [f & arg]
   (def args (first arg))
   ;(println "11111 "(first args))
   (letfn [(foo [result a s]
             (if (not-empty s)
               (do (println "a" a ", s" s ", result" result)
                   (foo (conj result  (apply f a (first args) result
                                             ;(if (and (not-empty result) (>(count result)1)) (last result))
                                             ))
                        (first s) (rest s)))
               result))]
     (foo [(first args)]  (first args) (rest args)))   ;[(first args)]
   ;([fa fb args] (do (println args) (reduce fa fb args)))
   )+ (range 5))
;=>[0 0 1 3 7]

((fn
   [f & arg]
   (def args (first arg))
   ;(println "11111 "(first args))
   (letfn [(foo [result a s]
             (if (not-empty s)
               (do (println "a" a ", s" s ", result" result)
                   (foo (conj result  (apply f a (first args) result
                                             ;(if (and (not-empty result) (>(count result)1)) (last result))
                                             ))
                        (first s) (rest s)))
               result))]
     (foo [(first args)]  (first args) (rest args)))   ;[(first args)]
   ;([fa fb args] (do (println args) (reduce fa fb args)))
   )+ (range 5))


(range 5);=>(0 1 2 3 4)
(last [9])
(= (take 5 (__ + (range))) [0 1 3 6 10])
(= (__ conj [1] [2 3 4]) [[1] [1 2] [1 2 3] [1 2 3 4]])
(= (last (__ * 2 [3 4 5])) (reduce * 2 [3 4 5]) 120)

;cheating
((fn reds  
  ([f coll]
   (lazy-seq
    (if-let [s (seq coll)]
      (reds f (first s) (rest s))
      (list (f)))))
  ([f init coll]
   (if (reduced? init)
     (list @init)
     (cons init
           (lazy-seq
            (when-let [s (seq coll)]
              (reds f (f init (first s)) (rest s))))))))* 2 [3 4 5])

(fn 
  ([fx collx]
   (letfn [(reds [f coll]
     (lazy-seq
      (if-let [s (seq coll)]
        (reds f (first s) (rest s))
        (list (f))))
  )]
   (reds fx collx)) )
  ([fx initx collx] 
   (letfn [(reds [f init coll]
    (if (reduced? init)
     (list @init)
      (cons init
       (lazy-seq
         (when-let [s (seq coll)]
           (reductions f (f init (first s)) (rest s)))))))]
     (reds fx initx collx))))

    (= (take 5 ((fn
                  ([fx collx]
                   (letfn [(reds [f coll]
                             (lazy-seq
                              (if-let [s (seq coll)]
                                (reds f (first s) (rest s))
                                (list (f)))))]
                     (reds fx collx)))
                  ([fx initx collx]
                   (letfn [(reds [f init coll]
                             (if (reduced? init)
                               (list @init)
                               (cons init
                                     (lazy-seq
                                      (when-let [s (seq coll)]
                                        (reductions f (f init (first s)) (rest s)))))))]
                     (reds fx initx collx)))) + (range))) [0 1 3 6 10])
    (= (__ conj [1] [2 3 4]) [[1] [1 2] [1 2 3] [1 2 3 4]])
    (= (last (__ * 2 [3 4 5])) (reduce * 2 [3 4 5]) 120)

((fn [& ss] (letfn [(reds [a b] (* a b))
                  (reds1 [a b c] (* a b c))]
            (let [s (list* ss)] 
              (println (count s))
            (if 
             (= (count s) 2)
              (reds s  )
              (reds1 s))) )     ) 7 6)
    
(seq '(7 9))

;; Write a function which behaves like reduce, but returns each intermediate value of the reduction. 
;; Your function must accept either two or three arguments, and the return sequence must be lazy.
;;     (= (take 5 (__ + (range))) [0 1 3 6 10])
;;     (= (__ conj [1] [2 3 4]) [[1] [1 2] [1 2 3] [1 2 3 4]])
;;     (= (last (__ * 2 [3 4 5])) (reduce * 2 [3 4 5]) 120)
;; Special Restrictions : reductions
    
(defn reductions
  "Returns a lazy seq of the intermediate values of the reduction (as
  per reduce) of coll by f, starting with init."
  {:added "1.2"}
  ([f coll]
   (lazy-seq
    (if-let [s (seq coll)]
      (reductions f (first s) (rest s))
      (list (f)))))
  ([f init coll]
   (if (reduced? init)
     (list @init)
     (cons init
           (lazy-seq
            (when-let [s (seq coll)]
              (reductions f (f init (first s)) (rest s))))))))

;++
(fn my-reduce
  ([f xs] (my-reduce f (first xs) (rest xs)))
  ([f a xs]
   (lazy-seq
    (if (empty? xs) (list a)
        (cons a (my-reduce f (f a (first xs)) (rest xs)))))))
;
(fn red
  ([f s] (lazy-seq (if-let [c (seq s)]
                     (red f (first c) (rest c)))))
  ([f i s] (cons i (lazy-seq (when-let [c (seq s)]
                               (red f (f i (first c)) (rest c)))))))

;--61.
;Map Construction
;; Write a function which takes a vector of keys and a vector of values and constructs a map from them.
;;     (= (__ [:a :b :c] [1 2 3]) {:a 1, :b 2, :c 3})
;;     (= (__ [1 2 3 4] ["one" "two" "three"]) {1 "one", 2 "two", 3 "three"})
;;     (= (__ [:foo :bar] ["foo" "bar" "baz"]) {:foo "foo", :bar "bar"})
;; Special Restrictions : zipmap
(fn [k v] (apply hash-map (reverse (interleave v k))))
(apply hash-map (reverse (interleave [:a :b :c] [1 2 3])))
;=>{1 :a, 3 :c, 2 :b}
(reverse {1 :a, 3 :c, 2 :b})
;=>([2 :b] [3 :c] [1 :a])
(= ((fn [k v] (apply hash-map (reverse (interleave v k)))) [:a :b :c] [1 2 3]) {:a 1, :b 2, :c 3})
(= ((fn [k v] (apply hash-map (reverse (interleave v k)))) [1 2 3 4] ["one" "two" "three"]) {1 "one", 2 "two", 3 "three"})
(= ((fn [k v] (apply hash-map (reverse (interleave v k)))) [:foo :bar] ["foo" "bar" "baz"]) {:foo "foo", :bar "bar"})

;--62.
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
(defn lazy
  ([]
   (concat [x] (lazy x)))
  ([x]
   (let [n (f x)]
     (lazy-seq
      (cons n (lazy n))))))
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
  (let [k (map f s)
        ks (set k)
        kv (vec ks)
        vs (vector ())]
            (zipmap ks vs)))

; from clojuredocs reduce
(defn transform
  [coll]
  (reduce (fn [ncoll [k v]]
            (assoc ncoll k (* 10 v)))
          {}
          coll))
(transform {:a 1 :b 2 :c 3})
;;{:a 10 :b 20 :c 30}
(transform {:a 1 :b 2 :c 3})
;;{:a 10 :b 20 :c 30}

; from clojuredocs reduce
(def x {:a 1 :b 2})
(reduce (fn [p [k v]]
          (into p {k (+ 1 v)}))
        {} ; First value for p
        x)
;; => {:a 2, :b 3}
(map (fn [a b] {a b}) (interleave (map #(> % 5) #{1 3 6 8}) #{1 3 6 8})
     (map hash-map {false 1 true 6 false 3 true 8})
     (interleave (map #(> % 5) #{1 3 6 8}) #{1 3 6 8})
;=>(false 1 true 6 false 3 true 8)
     (apply hash-map (interleave (map #(> % 5) #{1 3 6 8}) #{1 3 6 8}))
     (reduce-kv)
     (reduce)
     (assoc)
     (apply hash-map (interleave (map #(> % 5) #{1 3 6 8}) #{1 3 6 8}))
     (map hash-map (hash-map (false 1 true 6 false 3 true 8)))
     (map #(> % 5) #{1 3 6 8})
;=>(false true false true)
     (set (map #(> % 5) #{1 3 6 8}))
;=>#{true false}
     (vec (set (map #(> % 5) #{1 3 6 8})))
;=>[true false]
     (zipmap '(false true false true) (vec #{1 3 6 8}))
;=>{false 3, true 8}

     (map (fn [a b] {a [b]}) (map #(> % 5) #{1 3 6 8}) #{1 3 6 8})
;=>({false 1} {true 6} {false 3} {true 8})
     (apply merge-with into (map (fn [a b] {a [b]}) (map #(> % 5) #{1 3 6 8}) #{1 3 6 8}))
;=>{false [1 3], true [6 8]}
;works!!
     (fn [] (apply merge-with into (map (fn [a b] {a [b]}) k s)))
     (fn [f s]
       (let [k (map f s)]
         (apply merge-with into (map (fn [a b] {a [b]}) k s))))

     ((fn [f s]
        (let [k (map f s)]
          (apply merge-with into (map (fn [a b] {a [b]}) k s)))) #(> % 5) #{1 3 6 8})

;--66.
;Greatest Common Divisor
;; Given two integers, write a function which returns the greatest common divisor.
;; (= (__ 2 4) 2)
;; (= (__ 10 5) 5)
;; (= (__ 5 7) 1)
;; (= (__ 1023 858) 33)
     (fn [a b] (letfn [(foo [a b x]
                         (if (and (= (mod a x) 0)
                                  (= (mod b x) 0))
                           x
                           (foo a b (dec x))))]
                 (foo a b (min a b))))

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
     (complement  distinct)
     ((complement  distinct) (interleave #{0 1 2 3} #{2 3 4 5}))
     ((let [i (interleave #{0 1 2 3} #{2 3 4 5})]
        (defn f [s] ((complement  distinct) s))
        (println i)
        (f i)))
     (fn [s1 s2] (let [s (interleave s1 s2)])
       (dedupe s))
     distinct
     (hash-map (map #{0 1 2 3} #{2 3 4 5}))
;=>(nil 3 2 nil)
;Works!
     (set (filter #(not (nil? %)) (map #{0 1 2 3} #{2 3 4 5})))
     (fn [a b]
       (set (filter #(not (nil? %)) (map a b))))


     (condp)

     --
;--65.
;Black Box Testing
;; Clojure has many collection types, which act in subtly different ways. 
;; The core functions typically convert them into a uniform "sequence" type 
;; and work with them that way, but it can be important to understand the 
;; behavioral and performance differences so that you know which kind is 
;; appropriate for your application. Write a function which takes a collection 
;; and returns one of: map, :set, :list, or :vector - describing the type of 
;; collection it was given. You won't be allowed to inspect their class or use 
;; the built-in predicates like list? - the point is to poke at them and understand their behavior.
;;     (= :map (__ {:a 1, :b 2}))
;;     (= :list (__ (range (rand-int 20))))
;;     (= :vector (__ [1 2 3 4 5 6]))
;;     (= :set (__ #{10 (rand-int 5)}))
;;     (= [:map :set :vector :list] (map __ [{} #{} [] ()]))
;; Special Restrictions : class,type,Class,vector?,sequential?,list?,seq?,map?,set?,instance?,getClass
     ((try (into e p)
     (catch Exception ex :map)
     (finally
       (if flag
         (if (= (e p) #{1 2 3 4})
           :set
           (try
             (= (e p) (e p))
             (catch Exception ex1 :list)))))))
(fn [sx] (let [e (empty sx)
              p [1 2 3 4]]
          (letfn [(foo [s n]
             (case n
               0 (try
                   (into e p)
                   (foo s 1)
                   (catch Exception ex :map))
               1 (if (= (e p) #{1 2 3 4})
                   :set
                   (foo s 2))
               2 (try
                   (= (e p) (e p))
                   :vector
                   (catch Exception ex1 :list))
               ))]
            (foo sx 0)
                 )))
(= :map (__ {:a 1, :b 2}))
(= :list (__ (range (rand-int 20))))
(= :vector (__ [1 2 3 4 5 6]))
(= :set (__ #{10 (rand-int 5)}))
(= [:map :set :vector :list] (map __ [{} #{} [] ()]))
(type (keys [1 2 3 4 5 6])  )
(vals [1 2 3 4 5 6])
(try
     (seq [1 2 3 4 5 6])
     (catch Exception ex (if (not (nil? e)) (println "not") (println "e")))  
     (finally ))
(rest {:a 1, :b 2})
(empty (range (rand-int 20)))
(into (empty (range (rand-int 20))) [1 2])
;=>(2 1)
(into (empty  {:a 1, :b 2}) [1 2 3 4])
;=>; Execution error (IllegalArgumentException) at user/eval5641 (REPL:1478).
(into (empty #{10 (rand-int 5)})[1 2])
#{1 2}
(=(first [2 1]) (first '(2 1)))
(vector [2 1])
(='() '())
(counted? {:a 1, :b 2})
(associative?'(2 1))
((fn [sx] (let [e (empty sx)
                p [1 2 3 4]]
            (letfn [(foo [s n]
                      (case n
                        0 (try
                            (into e p)
                            (println "0."e)
                            (foo s 1)
                            (catch Exception ex 
                              :map))
                        1 (if (= (into e p) #{1 2 3 4})
                            (do
                             (println "1.2" e)
                            :set)
                            (do
                               (println "1.2" e)
                             (foo s 2)))
                        2 (try
                            (contains? e 0)
                            (println "2."e)
                            :vector
                            (catch Exception ex1 :list))))]
              (foo sx 0))))[1 2 3 4])
;works!    
(fn [sx] (let [e (empty sx)
               p [1 2 3 4]]
           (letfn [(foo [s n]
                     (case n
                       0 (try
                           (into e p)
                           (foo s 1)
                           (catch Exception ex
                             :map))
                       1 (if (= (into e p) #{1 2 3 4})
                           :set
                           (foo s 2))
                       2 (try
                           (contains? e 0)
                           :vector
                           (catch Exception ex1 :list))))]
             (foo sx 0))))
;Passed, but "Could not resolve symbol: Exception" error was raised
(keys {5 6})
(if-let [x (int? (keys '(5 6)))] x :not)
;=>:not
(if-let [x (int? (first (keys [5 6])))] :x :not)
;=>:not
(if-let [x (every? int? (keys {5 6}))] x :not)
;=>:not
(= :vector ((fn [sx] (let [e (empty sx)
                           p [1 2]]
                       (letfn [(foo [s n]
                                 (println  "e" e "p" p)
                                                           ;(println (into e [p]))
                                 (case n
                                   0 (if (= (into e [p]) {1 2})
                                       :map
                                       (foo s 1))
                                   1 (if (= (into e p) #{1 2})
                                       :set
                                       (foo s 2))
                                   2 (if (= (peek (into e p)) 2) :vector :list)))]
                         (foo sx 0)))) [1 2 3 4 5 6]))
(= [:map :set :vector :list] (map (fn [sx] (let [e (empty sx)
                                                 p [1 2]]
                                             (letfn [(foo [s n]
                                                       (println  "e" e "p" p)
                                                           ;(println (into e [p]))
                                                       (case n
                                                         0 (if (= (into e [p]) {1 2})
                                                             :map
                                                             (foo s 1))
                                                         1 (if (= (into e p) #{1 2})
                                                             :set
                                                             (foo s 2))
                                                         2 (if (= (peek (into e p)) 2) :vector :list)))]
                                               (foo sx 0)))) [{} #{} [] ()]))
(empty [{} #{} [] ()])
(quote [1 2])
;=>[1 2]    
(quote '(1 2))
???;=>(quote (1 2))
(quote {1 2})
;=>{1 2}
(quote #{1 2})
;=>#{1 2}

(fn [sx]
  (let [i (into (empty sx) [1 2])
        r (quote i)]
    (println "i:" i "r:")
    (case r
      [1 2] :vector
      {1 2} :map
      #{1 2} :set
      :list)))
(= :map ((fn [sx]
           (let [i (into  (empty sx) [[1 2]])
                 r (quote (into  (empty sx) [[1 2]]))]
             (println "i:" i "r:" r)
             (case i
               [[1 2]] :vector
               {1 2} :map
               #{[1 2]} :set
               :list))) {:a 1, :b 2}))
(= :list ((fn [sx]
            (let [i (into  (empty sx) [[1 2]])
                  r (quote i)]
              (println "i:" i "r:")
              (case i
                {1 2} :map
                #{[1 2]} :set
                (if (= (quote i) [[1 2]])
                  :vector
                  :list)))) (range (rand-int 20))))
(= :vector ((fn [sx]
              (let [i (into  (empty sx) [[1 2]])
                    r (quote i)]
                (println "i:" i "r:")
                (case i
                  {1 2} :map
                  #{[1 2]} :set
                  (if (= (quote i) [[1 2]])
                    :vector
                    :list)))) [1 2 3 4 5 6]))
(= :set (__ #{10 (rand-int 5)}))
(= [:map :set :vector :list] (map __ [{} #{} [] ()]))

(peek  [1 2])
(peek (list 1 2))
(peek (into '() [8 7]))
(peek (into [] [8 7]))
(assoc [8 7] 0 0)
(first (assoc '(8 7) 0 0))
(vector [7 8])
(vec [7 8])
(vector '(7 8))
(vec '(7 8))
(= (vector '(7 8)) (vector [7 8]));=> true
(= (vector '([1 2])) [[[1 2]]]);=> true!!!!!
;works, but why? 
(fn [sx] (let [e (empty sx)
               p [1 2]]
           (letfn [(foo [s n]
                     (println  "e" e "p" p)
                          ;(println (into e [p]))
                     (case n
                       0 (if (= (into e [p]) {1 2}) :map
                             (foo s 1))
                       1 (if (= (into e p) #{1 2})  :set
                             (foo s 2))
                       2 (if (= (vector (into e p)) [[1 2]]) :vector :list)))]
             (foo sx 0))))

(identical? '(1 2) [1 2]);=>false    
(constantly)

; works!!!
(fn [sx] (let [e (empty sx)
               p [1 2]]
           (if (= (into e [p]) {1 2}) :map
               (if (= (into e p) #{1 2}) :set
                   (if (=  (into e p) [1 2]) :vector :list)))))


;--88.
;Symmetric Difference
;; Write a function which returns the symmetric difference of two sets. 
;; The symmetric difference is the set of items belonging to one but not both of the two sets.
;; (= (__ #{1 2 3 4 5 6} #{1 3 5 7}) #{2 4 6 7})
;; (= (__ #{:a :b :c} #{}) #{:a :b :c})
;; (= (__ #{} #{4 5 6}) #{4 5 6})
;; (= (__ #{[1 2] [2 3]} #{[2 3] [3 4]}) #{[1 2] [3 4]})
(fn [a b] (set (map first (filter #(= (second %) 1) (frequencies (concat a b))))))
(fn [a b]
  (let [s (map b a)]
    (zipmap s a)))
(concat #{1 2 3 4 5 6} #{1 3 5 7})
;=>(1 4 6 3 2 5 7 1 3 5)
(frequencies (concat #{1 2 3 4 5 6} #{1 3 5 7}))
;=>{1 2, 4 1, 6 1, 3 2, 2 1, 5 2, 7 1}
(filter   #(= (second %) 1) (frequencies (concat #{1 2 3 4 5 6} #{1 3 5 7})))
;=>([4 1] [6 1] [2 1] [7 1])
(set (map first (filter   #(= (second %) 1) (frequencies (concat #{1 2 3 4 5 6} #{1 3 5 7})))))
((fn [a b]
   (let [s (map (contains? b) a)]
     (zipmap a s))) #{1 3 5 7} #{1 2 3 4 5 6})
(frequencies)
(map #{1 3 5 7} #{1 2 3 4 5 6})
;(1 nil nil 3 nil 5)
;works
(fn [a b] (set (map first (filter #(= (second %) 1) (frequencies (concat a b))))))

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
(map first (tree-seq next rest '(:a (:b nil nil) nil)))
;=>(:a :b nil nil nil)
(fn [ss] (letfn [(is-tree? [s] (and
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (or nil? (is-tree?) (second s))
                                (or nil? (is-tree?) (last s))
                            ;; (if (every? #(not (sequential? %)) s)
                            ;;  (or nil? (sequential? (second s)))   
                            ;;  (or nil? (sequential? (last s))))  
                                ))](is-tree? ss)))
(count '(:a (:b nil nil) nil))
;https://clojuredocs.org/clojure.core/tree-seq
;; The nodes are filtered based on their collection type, 
;; here they must be a list.
(tree-seq seq? seq [[1 2 [3]] [4]])
;;=> ([[1 2 [3]] [4]])
;; notice the difference between seq? and sequential?
(tree-seq sequential? seq [[1 2 [3]] [4]])
;;=> ([[1 2 [3]] [4]] [1 2 [3]] 1 2 [3] 3 [4] 4)

;; this is kind of weird IMO... but it works that way (the same for vectors)
;; See: https://en.wikipedia.org/wiki/Vacuous_truth
user=> (every? true? '())
true
user=> (every? false? '())
true
;; and similarly
user=> (every? map? '())
true
user=> (every? vector? '())
true
user=> (every? string? '())
true
user=> (every? number? '())
true
;; As such a better description of every? would be
;; Returns false if there exists a value x in coll 
;; such that (pred? x) is false, else true." 
...
;; however, invoking a set with a value returns the matched element,
;; causing the comparison below to fail
(subset? #{true false} #{true false}) ;;=> true
(every?  #{true false} #{true false}) ;;=> false ✘ 😦
(fn [ss] (letfn [(is-tree? [s] (and
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (#(or (nil? %) (= (empty s) %) (is-tree? %)) (second s))
                                ((or nil? #(= (empty s) %) is-tree?) (last s))
                            ;; (if (every? #(not (sequential? %)) s)
                            ;;  (or nil? (sequential? (second s)))   
                            ;;  (or nil? (sequential? (last s))))  
                                ))](is-tree? ss)))
;works
(fn [ss] (letfn [(is-tree? [s] (and
                                (sequential? s)
                                (boolean (not-empty s))
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (#(or (nil? %) (is-tree? %)) (second s))
                                (#(or (nil? %)  (is-tree? %)) (last s))))] (is-tree? ss)))
((or nil? is-tree?) (second s))
(and (print 8) true)
;=> nil
(seq? '())
(= nil false)
(nil? nil)
(def s ())
(and (:a nil ()) true)
(#(nil? %) (last (:a nil ())))

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
                                (#(or (nil? %)  (is-tree? %)) (last s))))]
           [(reverse-tree [res s]
                          (if (is-tree? s)
            ;;  (and
            ;;      (sequential? s) 
            ;;      (boolean (not-empty s)))              
                            res))]
           (is-tree? ss)))

(fn [s] (letfn [(is-tree? [s] (and
                               (sequential? s)
                               (boolean (not-empty s))
                               (= (count s) 3)
                               (not (sequential? (first s)))
                               (#(or (nil? %) (is-tree? %)) (second s))
                               (#(or (nil? %)  (is-tree? %)) (last s))))]

          (and
           (is-tree? s)
           (= (second s) (last s)))))

(= ((fn [s] (letfn [(is-tree? [s] (and
                                   (sequential? s)
                                   (boolean (not-empty s))
                                   (= (count s) 3)
                                   (not (sequential? (first s)))
                                   (#(or (nil? %) (is-tree? %)) (second s))
                                   (#(or (nil? %)  (is-tree? %)) (last s))))]

              (and
               (is-tree? s)
               (= (second s) (last s))))) '(:a (:b nil nil) (:b nil nil))) true)
(= ((fn [s] (letfn [(is-tree? [s] (and
                                   (sequential? s)
                                   (boolean (not-empty s))
                                   (= (count s) 3)
                                   (not (sequential? (first s)))
                                   (#(or (nil? %) (is-tree? %)) (second s))
                                   (#(or (nil? %)  (is-tree? %)) (last s))))]

              (and
               (is-tree? s)
               (= (second s) (last s))))) '(:a (:b nil nil) nil)) false)
(= ((fn [s] (letfn [(is-tree? [s] (and
                                   (sequential? s)
                                   (boolean (not-empty s))
                                   (= (count s) 3)
                                   (not (sequential? (first s)))
                                   (#(or (nil? %) (is-tree? %)) (second s))
                                   (#(or (nil? %)  (is-tree? %)) (last s))))]

              (and
               (is-tree? s)
               (= (second s) (last s))))) '(:a (:b nil nil) (:c nil nil))) false)

(def a [2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]])
(def b [2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil])

(flatten a)
;=>(2 nil 3 4 5 nil nil 6 nil nil nil)
(flatten b)
;=>(2 3 nil 4 6 nil nil 5 nil nil nil)
(tree-seq sequential? identity a)
;=>([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] 2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(tree-seq sequential? identity b)
;=>([2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil] 2 [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil nil)
(tree-seq seq? identity a)
;=>([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]])
(tree-seq seq? identity b)
;=>([2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil])
(= a b)
(tree-seq seq? reverse a)
;=>([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]])
(tree-seq sequential? reverse a)
;=>([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] [3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3 nil 2)
(tree-seq sequential? identity a)
;=>([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] 2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(rest (tree-seq sequential? identity a))
(2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(rest (tree-seq sequential? reverse a))
([3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3 nil 2)
(rest (tree-seq sequential? identity b))
(2 [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil nil)
(rest (tree-seq sequential? reverse b))
(nil [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 2)
seqable?

(rest (tree-seq seqable? identity a))
(2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(rest (tree-seq seqable? reverse a))
([3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3 nil 2)
(rest (tree-seq seqable? identity b))
(2 [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil nil)
(rest (tree-seq seqable? reverse b))
(nil [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 2)

(flatten a)
(2 nil 3 4 5 nil nil 6 nil nil nil)
(flatten b)
(2 3 nil 4 6 nil nil 5 nil nil nil)

(flatten (reverse a))
(3 4 5 nil nil 6 nil nil nil nil 2)
(flatten (reverse b))
(nil 3 nil 4 6 nil nil 5 nil nil 2)

(cons 1 [8 9])
;=>(1 8 9)

(fn [ss] (letfn [(is-tree? [s] (and
                                (sequential? s)
                                (boolean (not-empty s))
                                (= (count s) 3)
                                (not (sequential? (first s)))
                                (#(or (nil? %) (is-tree? %)) (second s))
                                (#(or (nil? %)  (is-tree? %)) (last s))))
                 (rev [res s]
                   (if (sequential? s)))]

           (and
            (is-tree? ss)
            (= (second ss) (last ss)))))


a
[2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]]
b
[2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil]
(defn rev [s] (if (sequential? s) (cons (first s) (reverse (rest s))) s))

(tree-seq sequential? #(cons (first %) (last %)) a)
([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] 2 3 [4 [5 nil nil] [6 nil nil]] 4 6 nil nil nil)
(tree-seq sequential? #(cons (first %) (rev (last %))) (rev a))
([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] 2 nil [4 [5 nil nil] [6 nil nil]] 4 nil nil 6 3)
(tree-seq sequential? #(cons (first %) (last %)) b)
([2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil] 2)
(tree-seq sequential? #(cons (last %) (rev (first %))) (rev b))
((nil [3 nil [4 [6 nil nil] [5 nil nil]]] 2) 2)

--
(tree-seq sequential? rev a)
([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] [3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3 nil 2)
(tree-seq sequential? rev (reverse a))
(([3 [4 [5 nil nil] [6 nil nil]] nil] nil 2) 2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3)
(tree-seq sequential? identity a)
([2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]] 2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(tree-seq sequential? identity (reverse a))
(([3 [4 [5 nil nil] [6 nil nil]] nil] nil 2) [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil nil 2)

(tree-seq sequential? rev b)
([2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil] nil [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 2)
(tree-seq sequential? rev (reverse b))
((nil [3 nil [4 [6 nil nil] [5 nil nil]]] 2) 2 [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 nil)
(tree-seq sequential? identity b)
([2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil] 2 [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil nil)
(tree-seq sequential? identity (reverse b))
((nil [3 nil [4 [6 nil nil] [5 nil nil]]] 2) nil [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil 2)

-
(rest (tree-seq sequential? rev a))
([3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3 nil 2)
(rest (tree-seq sequential? rev (reverse a)))
(2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] nil [4 [5 nil nil] [6 nil nil]] [6 nil nil] nil nil 6 [5 nil nil] nil nil 5 4 3)
(rest (tree-seq sequential? identity a))
(2 nil [3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil)
(rest (tree-seq sequential? identity (reverse a)))
([3 [4 [5 nil nil] [6 nil nil]] nil] 3 [4 [5 nil nil] [6 nil nil]] 4 [5 nil nil] 5 nil nil [6 nil nil] 6 nil nil nil nil 2)

(rest (tree-seq sequential? rev b))
(nil [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 2)
(rest (tree-seq sequential? rev (reverse b)))
(2 [3 nil [4 [6 nil nil] [5 nil nil]]] [4 [6 nil nil] [5 nil nil]] [5 nil nil] nil nil 5 [6 nil nil] nil nil 6 4 nil 3 nil)
(rest (tree-seq sequential? identity b))
(2 [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil nil)
(rest (tree-seq sequential? identity (reverse b)))
(nil [3 nil [4 [6 nil nil] [5 nil nil]]] 3 nil [4 [6 nil nil] [5 nil nil]] 4 [6 nil nil] 6 nil nil [5 nil nil] 5 nil nil 2)

(defn rests [s] (if (sequential? s) (rest s) s))
(defn firsts [s] (if (sequential? s) (first s) s))

(map rests (tree-seq sequential? rev b))
(([3 nil [4 [6 nil nil] [5 nil nil]]] nil) nil (nil [4 [6 nil nil] [5 nil nil]]) ([6 nil nil] [5 nil nil]) (nil nil) nil nil 5 (nil nil) nil nil 6 4 nil 3 2)
(map rests (tree-seq sequential? rev (reverse b)))
(([3 nil [4 [6 nil nil] [5 nil nil]]] 2) 2 (nil [4 [6 nil nil] [5 nil nil]]) ([6 nil nil] [5 nil nil]) (nil nil) nil nil 5 (nil nil) nil nil 6 4 nil 3 nil)
(map rests (tree-seq sequential?  identity b))
(([3 nil [4 [6 nil nil] [5 nil nil]]] nil) 2 (nil [4 [6 nil nil] [5 nil nil]]) 3 nil ([6 nil nil] [5 nil nil]) 4 (nil nil) 6 nil nil (nil nil) 5 nil nil nil)
(map rests (tree-seq sequential? identity (reverse b)))
(([3 nil [4 [6 nil nil] [5 nil nil]]] 2) nil (nil [4 [6 nil nil] [5 nil nil]]) 3 nil ([6 nil nil] [5 nil nil]) 4 (nil nil) 6 nil nil (nil nil) 5 nil nil 2)

(map firsts (tree-seq sequential? rev b)) ;!!!!!!
(2 nil 3 4 5 nil nil 5 6 nil nil 6 4 nil 3 2)
(map firsts (tree-seq sequential? rev (reverse b)))
(nil 2 3 4 5 nil nil 5 6 nil nil 6 4 nil 3 nil)
(map firsts (tree-seq sequential?  identity b))
(2 2 3 3 nil 4 4 6 6 nil nil 5 5 nil nil nil)
(map firsts (tree-seq sequential? identity (reverse b)))
(nil nil 3 3 nil 4 4 6 6 nil nil 5 5 nil nil 2)

(map firsts (tree-seq sequential? rev a)) ;!!!!!!
(2 3 nil 4 6 nil nil 6 5 nil nil 5 4 3 nil 2)
(map firsts (tree-seq sequential? rev (reverse a)))
(nil 2 3 4 5 nil nil 5 6 nil nil 6 4 nil ([3 [4 [5 nil nil] [6 nil nil]] nil] 2 nil 3 nil 4 6 nil nil 6 5 nil nil 5 4 3)
     3 nil)
(map firsts (tree-seq sequential?  identity a))
(2 2 nil 3 3 4 4 5 5 nil nil 6 6 nil nil nil)
(map firsts (tree-seq sequential? identity (reverse a)))
([3 [4 [5 nil nil] [6 nil nil]] nil] 3 3 4 4 5 5 nil nil 6 6 nil nil nil nil 2)

--
(map firsts (tree-seq sequential? rev b));!!!!!!
(2 2 nil 3 3 4 4 5 5 nil nil 6 6 nil nil nil)
(map firsts (tree-seq sequential? rev (reverse b)))
(nil nil 2 3 3 4 4 5 5 nil nil 6 6 nil nil nil)
(map firsts (tree-seq sequential?  identity b));!!!
(2 2 3 3 nil 4 4 6 6 nil nil 5 5 nil nil nil)
(map firsts (tree-seq sequential? identity (reverse b)))
(nil nil 3 3 nil 4 4 6 6 nil nil 5 5 nil nil 2)

(map firsts (tree-seq sequential? rev a)) ;!!!
(2 2 3 3 nil 4 4 6 6 nil nil 5 5 nil nil nil)
(map firsts (tree-seq sequential? rev (reverse a)))
([3 [4 [5 nil nil] [6 nil nil]] nil] 3 3 nil 4 4 6 6 nil nil 5 5 nil nil 2 nil)
(map firsts (tree-seq sequential?  identity a)) ;!!!!!!
(2 2 nil 3 3 4 4 5 5 nil nil 6 6 nil nil nil)
(map firsts (tree-seq sequential? identity (reverse a)))
([3 [4 [5 nil nil] [6 nil nil]] nil] 3 3 4 4 5 5 nil nil 6 6 nil nil nil nil 2)


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
            (= (map firsts (tree-seq sequential? rev a))
               (reverse (map firsts (tree-seq sequential? rev b)))))))

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
            (= (map firsts (tree-seq sequential? rev (second ss)))
               (reverse (map firsts (tree-seq sequential? rev (last ss))))))))

(= ((fn [ss] (letfn [(is-tree? [s] (and
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
                   (map firsts (tree-seq sequential? rev (last ss))))))) '(:a (:b nil nil) (:b nil nil))) true)
(def ss [1 [2 nil [3 [4 [5 nil nil] [6 nil nil]] nil]]
         [2 [3 nil [4 [6 nil nil] [5 nil nil]]] nil]]
  ;'(:a (:b nil nil) (:b nil nil))
  )
(defn is-tree? [s] (and
                    (sequential? s)
                    (boolean (not-empty s))
                    (= (count s) 3)
                    (not (sequential? (first s)))
                    (#(or (nil? %) (is-tree? %)) (second s))
                    (#(or (nil? %)  (is-tree? %)) (last s))))
(is-tree? ss)
(= (map firsts (tree-seq sequential? rev (second ss)))
   (reverse (map firsts (tree-seq sequential? rev (last ss)))))

(map firsts (tree-seq sequential? rev (second ss)))
(:b :b nil nil)
(reverse (map firsts (tree-seq sequential? rev (reverse (last  ss)))))
(nil nil :b :b)
; works!!!
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
                   (if (= m (+ n 1))
                     prev
                     (foo (concat 1 (reduced + prev) 1) (inc m))))]
          ;;  (case n
          ;;   1 [1]
          ;;   (foo [1]))
           (into [] (foo [1])))))

((fn ([n] (letfn [(foo [prev m]
                    ;(println prev m)   
                    (if (= m (+ n 1))
                      prev
                      (foo (concat '(1) (red prev) '(1)) (inc m))))
                  (red [s] (map #(apply + %)
                                (partition 2
                                           (concat
                                            (list (first s))
                                            (mapcat #(repeat 2 %) (rest (butlast s)))
                                            (list (last s))))))]
            (into [] (foo '(1) 2))))) 2)
;works!
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
; I liked :)

;--99.
;Product Digits
;; Write a function which multiplies two numbers and returns the result as a sequence of its digits.
;; (= (__ 1 1) [1])
;; (= (__ 99 9) [8 9 1])
;; (= (__ 999 99) [9 8 9 0 1])
chars
(str \9)
(-> \9 (str) (Integer/parseInt))
;=>9
(num \1)
(fn [a b] (mapv #(-> % (str) (Integer/parseInt)) (seq (str (* a b)))))
((fn [a b] (mapv #(-> % (str) (Integer/parseInt)) (seq (str (* a b))))) 99 9)
;#error {:message "Could not resolve symbol: Integer/parseInt", :data {:type :sci/error, :line 1, :column 35, :file nil, :phase "analysis"}}
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
            [(nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
                  (cond 
                    (= a 0) b
                    (= b 0) b
                    (= a b) a
                    (and (even? a) (even? b)) (* 2 (nod (/ a 2)(/ b 2)))
                    (and (even? a) (odd? b)) (nod (/ a 2) b)
                    (and (even? b) (odd? a)) (nod (/ b 2) a)
                    (and (odd? b) (odd? a) (> a b)) (nod (/ (- a b) 2) b)
                    (and (odd? b) (odd? a) (< a b)) (nod (/ (- b a) 2) a)))
             (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
                  (/ (* a b) (nod a b)) )]
             (reduce nok ss))
  )
;https://ru.wikipedia.org/wiki/%D0%9D%D0%B0%D0%B8%D0%BC%D0%B5%D0%BD%D1%8C%D1%88%D0%B5%D0%B5_%D0%BE%D0%B1%D1%89%D0%B5%D0%B5_%D0%BA%D1%80%D0%B0%D1%82%D0%BD%D0%BE%D0%B5
;https://ru.wikipedia.org/wiki/%D0%90%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%95%D0%B2%D0%BA%D0%BB%D0%B8%D0%B4%D0%B0
;https://ru.wikipedia.org/wiki/%D0%91%D0%B8%D0%BD%D0%B0%D1%80%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC_%D0%B2%D1%8B%D1%87%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F_%D0%9D%D0%9E%D0%94
;https://ru.wikipedia.org/wiki/%D0%9D%D0%B0%D0%B8%D0%B1%D0%BE%D0%BB%D1%8C%D1%88%D0%B8%D0%B9_%D0%BE%D0%B1%D1%89%D0%B8%D0%B9_%D0%B4%D0%B5%D0%BB%D0%B8%D1%82%D0%B5%D0%BB%D1%8C
;works for int
(fn [& ss] (letfn
            [(nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
               (cond
                 (= a 0) b
                 (= b 0) b
                 (= a b) a
                 (and (even? a) (even? b)) (* 2 (nod (/ a 2) (/ b 2)))
                 (and (even? a) (odd? b)) (nod (/ a 2) b)
                 (and (even? b) (odd? a)) (nod (/ b 2) a)
                 (and (odd? b) (odd? a) (> a b)) (nod (/ (- a b) 2) b)
                 (and (odd? b) (odd? a) (< a b)) (nod (/ (- b a) 2) a)))
             (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
               (/ (* a b) (nod a b)))]
             (reduce nok ss)))

(fn [& ss] (letfn
                 [(nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
                    (cond
                      (= a 0) b
                      (= b 0) b
                      (= a b) a
                      (and (even? a) (even? b)) (* 2 (nod (/ a 2) (/ b 2)))
                      (and (even? a) (odd? b)) (nod (/ a 2) b)
                      (and (even? b) (odd? a)) (nod (/ b 2) a)
                      (and (odd? b) (odd? a) (> a b)) (nod (/ (- a b) 2) b)
                      (and (odd? b) (odd? a) (< a b)) (nod (/ (- b a) 2) a)))
                  (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
                    (/ (* a b) (nod a b)))]
                  (if (some ratio? ss) 
                   (let [denominators-nok (reduce nok (map denominator ss))
                         numerators (map #(* (numerator %) (/ denominators-nok (denominator %))) ss)
                         numerators-nok (reduce nok numerators)]  
                     (rationalize (/ numerators-nok denominators-nok))   
                     (reduce nok ss))))) 
(== ((fn [& ss] (letfn
                 [(nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
                    (cond
                      (= a 0) b
                      (= b 0) b
                      (= a b) a
                      (and (even? a) (even? b)) (* 2 (nod (/ a 2) (/ b 2)))
                      (and (even? a) (odd? b)) (nod (/ a 2) b)
                      (and (even? b) (odd? a)) (nod (/ b 2) a)
                      (and (odd? b) (odd? a) (> a b)) (nod (/ (- a b) 2) b)
                      (and (odd? b) (odd? a) (< a b)) (nod (/ (- b a) 2) a)))
                  (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
                    (/ (* a b) (nod a b)))]
                  (reduce nok ss))) 5 3 7) 105)
;works for fractions
(fn [& ss] (letfn
            [(nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
               (cond
                 (= a 0) b
                 (= b 0) b
                 (= a b) a
                 (and (even? a) (even? b)) (* 2 (nod (/ a 2) (/ b 2)))
                 (and (even? a) (odd? b)) (nod (/ a 2) b)
                 (and (even? b) (odd? a)) (nod (/ b 2) a)
                 (and (odd? b) (odd? a) (> a b)) (nod (/ (- a b) 2) b)
                 (and (odd? b) (odd? a) (< a b)) (nod (/ (- b a) 2) a)))
             (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
               (/ (* a b) (nod a b)))]
             (if (some ratio? ss)
               (let [denominators-nok (reduce nok (map #(denominator %) ss))
                     numerators (map #(* (numerator %) (/ denominators-nok (denominator %))) ss)
                     numerators-nok (reduce nok numerators)]
                 (rationalize (/ numerators-nok denominators-nok)))
               (reduce nok ss))))
;works!!! for ints and for fractions 
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

(== ((fn [& ss] (letfn
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
                    (reduce nok ss)))) 7 5/7 2 3/5) 210)
(even? 5N)
(map #(denominator %) [1/3 2/5])
(some ratio?[1/3 2/5] )
(/ 5 6N)
(ratio 4)
(if-not (ratio? %) (/ (* % 2) 2))9
(odd? 3/4)
;; #error {:message "EOF while reading, expected ) to match ( at [1,5]", :data {:type :sci.error/parse, :line 25, :column 47, 
;; :edamame/expected-delimiter ")", :edamame/opened-delimiter "(", :edamame/opened-delimiter-loc {:row 1, :col 5}, :phase "parse",
;; :file nil}, :cause #error {:message "EOF while reading, expected ) to match ( at [1,5]", :data {:type :edamame/error, :line 25, 
;; :column 47, :edamame/expected-delimiter ")", :edamame/opened-delimiter "(", :edamame/opened-delimiter-loc {:row 1, :col 5}}}}

;doesnt work
;; (fn [& ss] (letfn
;;                  [(r-even? [n] (if (int? n) 
;;                                  (even? n)
;;                                  (and (even? (numerator n)) (even? (denominator n)))))
;;                   (r-odd? [n] (complement r-even?))
;;                   (nod [a b] ;https://en.wikipedia.org/wiki/Binary_GCD_algorithm
;;                     (cond
;;                       (= a 0) b
;;                       (= b 0) b
;;                       (= a b) a
;;                       (and (r-even? a) (r-even? b)) (* 2 (nod (/ a 2) (/ b 2)))
;;                       (and (r-even? a) (r-odd? b)) (nod (/ a 2) b)
;;                       (and (r-even? b) (r-odd? a)) (nod (/ b 2) a)
;;                       (and (r-odd? b) (r-odd? a) (> a b)) (nod (/ (- a b) 2) b)
;;                       (and (r-odd? b) (r-odd? a) (< a b)) (nod (/ (- b a) 2) a)))
;;                   (nok [a b] ;https://en.wikipedia.org/wiki/Least_common_multiple
;;                     (/ (* a b) (nod a b)))]
;;                   (if (every? int? ss)
;;                     (reduce nok ss)
;;                     (let [denominators-nok (reduce nok (map #(if (int? %) 1 (denominator %) ) ss))
;;                           numerators (map #(* (if (int? %) % (numerator %)) 
;;                                               (/ denominators-nok 
;;                                                  (if (int? %) 1 (denominator %) ))) ss)
;;                           numerators-nok (reduce nok numerators)]
;;                       (rationalize (/ numerators-nok denominators-nok)))
;;                     )))
;user=> "#error {:message \"Could not resolve symbol: numerator\", 
;:data {:type :sci/error, :line 4, :column 47, :file nil, :phase \"analysis\"}}"

;https://4clojure.oxal.org/#/problem/100
;breaks
(fn [& x]
   (let [y (apply min x)]
     (loop [z y]
       (if (every? #(zero? (mod z %)) x)
         z
         (recur (+ z y))))))
;works
(fn [e & r]
  ((fn f [p]
     (if (every? #(= (mod p %) 0) r)
       p
       (f (+ p e)))) e))
(mod 18 18)

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
(fn [n] (fn [x] (int (Math/pow x n))))

;https://stackoverflow.com/questions/5057047/how-to-do-exponentiation-in-clojure
(require '[clojure.math.numeric-tower :as math])
(math/expt 4 2)
;=> 16
or
(require '[clojure.math.numeric-tower :as math :refer [expt]])
(expt 4 2)
;=> 16
(clojure.string/join ["4" "2"]); =>"42"
;doesnt work (clojure.math.numeric-tower/expt )
(Math/pow 4 2)

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

(fn [f ss]
  (letfn
   [(mapr [res s]
      (if (not (empty? s))
        (mapr (conj res (f (first s))) (rest s))
        res))]
    (mapr (empty ss) ss)))

(= [3 4 5 6 7]
   ((fn [f ss]
      (letfn
       [(mapr [res s]
          (if (not (empty? s))
            (mapr (conj res (f (first s))) (rest s))
            res))]
        (mapr (empty ss) ss))) inc [2 3 4 5 6]))
(= (repeat 10 nil)
   ((fn [f ss]
      (letfn
       [(mapr [res s]
          (if (not (empty? s))
            (mapr (conj res (f (first s))) (rest s))
            res))]
        (lazy-seq (mapr (empty ss) ss)))) (fn [_] nil) (range 10)))
(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [res s]
               (if (not (empty? s))
                 (mapr (conj res (f (first s))) (rest s))
                 res))]
             (mapr (lazy-seq (empty ss)) ss))) inc (range))
        (drop (dec 1000000))
        (take 2)))
; Execution error (StackOverflowError) at user/eval8438$fn$mapr (REPL:581).
; null

(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [res s]
               (if (empty? s)
                 res
                 (mapr
                  (lazy-seq (conj res
                                  (f (first s))))
                  (lazy-seq (rest s)))))]
             (mapr  (empty ss)  (lazy-seq ss)))) inc (range))
        (drop (dec 1000000))
        (take 2)))
; Execution error (OutOfMemoryError) at user/eval7441$fn$mapr (REPL:593).
; Java heap space

(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [res s]
               (if (empty? s)
                 res
                 (recur (conj res (f (first s))) (rest s))))]
             (mapr  (empty ss) (lazy-seq ss)))) inc (range))
        (drop (dec 1000000))
        (take 2)))

((fn [f ss]
   (letfn
    [(mapr [res s]
       (if (not (empty? s))
         (mapr (conj res (f (first s))) (rest s))
         res))]
     (mapr (empty ss) ss))) inc [2 3 4 5 6])
(conj (lazy-seq []) 7)
(cons)

(fn [f ss]
  (letfn
   [(mapr [s]
      (lazy-seq (while (not (empty? s))
                  (conj (f (first s))) (rest s))))]
    (mapr  (lazy-seq ss))))
(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [s]
               (lazy-seq
                (conj
                 (f (first s)))
                (mapr (rest s))))]
             (mapr   ss))) inc (range 1000008))
        (drop (dec 1000000))
        (take 2)))
; Execution error (NullPointerException) at user/eval5702$fn$mapr$fn (REPL:640).
; null

;works!
(fn [f ss]
  (letfn
   [(mapr [s]
      (lazy-seq
       (if (not (empty? s))
         (cons
          (f (first s))
          (mapr (rest s))))))]
    (mapr   ss)))

(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [s]
               (lazy-seq
                (if (not (empty? s))
                  (cons
                   (f (first s))
                   (mapr (rest s))))))]
             (mapr   ss))) inc (range))
        (drop (dec 1000000))
        (take 2)))


(= [3 4 5 6 7]
   ((fn [f ss]
      (letfn
       [(mapr [s]
          (lazy-seq
           (if (not (empty? s))
             (cons
              (f (first s))
              (mapr (rest s))))))]
        (mapr   ss))) inc [2 3 4 5 6]))
(= (repeat 10 nil)
   ((fn [f ss]
      (letfn
       [(mapr [s]
          (lazy-seq
           (if (not (empty? s))
             (cons
              (f (first s))
              (mapr (rest s))))))]
        (mapr   ss))) (fn [_] nil) (range 10)))
(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [s]
               (lazy-seq
                (cons
                 (f (first s))
                 (mapr (rest s)))))]
             (mapr   ss))) inc (range))
        (drop (dec 1000000))
        (take 2)))
(= [1000000 1000001]
   (->> ((fn [f ss]
           (letfn
            [(mapr [s]
               (lazy-seq
                (cons
                 (f (first s))
                 (mapr (rest s)))))]
             (mapr  (lazy-seq ss)))) inc (range))
        (drop (dec 1000000))
        (take 2)))
;Maximum number of elements realized: 10000

;Clojure.core
(defn map
  "Returns a lazy sequence consisting of the result of applying f to
  the set of first items of each coll, followed by applying f to the
  set of second items in each coll, until any one of the colls is
  exhausted.  Any remaining items in other colls are ignored. Function
  f should accept number-of-colls arguments. Returns a transducer when
  no collection is provided."
  {:added "1.0"
   :static true}
  ([f]
   (fn [rf]
     (fn
       ([] (rf))
       ([result] (rf result))
       ([result input]
        (rf result (f input)))
       ([result input & inputs]
        (rf result (apply f input inputs))))))
  ([f coll]
   (lazy-seq
    (when-let [s (seq coll)]
      (if (chunked-seq? s)
        (let [c (chunk-first s)
              size (int (count c))
              b (chunk-buffer size)]
          (dotimes [i size]
            (chunk-append b (f (.nth c i))))
          (chunk-cons (chunk b) (map f (chunk-rest s))))
        (cons (f (first s)) (map f (rest s)))))))
  ([f c1 c2]
   (lazy-seq
    (let [s1 (seq c1) s2 (seq c2)]
      (when (and s1 s2)
        (cons (f (first s1) (first s2))
              (map f (rest s1) (rest s2)))))))
  ([f c1 c2 c3]
   (lazy-seq
    (let [s1 (seq c1) s2 (seq c2) s3 (seq c3)]
      (when (and  s1 s2 s3)
        (cons (f (first s1) (first s2) (first s3))
              (map f (rest s1) (rest s2) (rest s3)))))))
  ([f c1 c2 c3 & colls]
   (let [step (fn step [cs]
                (lazy-seq
                 (let [ss (map seq cs)]
                   (when (every? identity ss)
                     (cons (map first ss) (step (map rest ss)))))))]
     (map #(apply f %) (step (conj colls c3 c2 c1))))))

https://4clojure.oxal.org/#/problem/118/solutions
(fn m [f c] (lazy-seq (when-let [[x & y] (seq c)] (cons (f x) (m f y)))))
(fn [f ss]
  (letfn
   [(mapr [s]
      (lazy-seq
       (if (not (empty? s))
         (cons
          (f (first s))
          (mapr (rest s))))))]; 
    (mapr   ss))); or (mapr   (lazy-seq ss))))
;error Maximum number of elements realized: 10000
;mine final
(fn [f s]
  (lazy-seq
   (if  (seq s)
     (cons
      (f (first s))
      (map (rest s))))))
(lazy-cat)
;https://4clojure.oxal.org/#/problem/118/solutions
(fn m [f [x & xs :as s]]
  (if (empty? s)
    []
    (lazy-seq (cons (f x) (m f xs)))))
(fn m [f [h & r]] (lazy-seq (cons (f h) (if r (m f r)))))

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
;
(fn  [ss]
  (letfn [(bin [s n]
            (while (seq s)
              (let [d (read-string (str (last s)))]
                (+ (if-not (= d 0) (reduce * (repeat n 2)))
                   (bin (butlast s) (inc n))))))]
    (bin ss 0)))
(= 0 ((fn a [ss]
        (letfn [(bin [s n]
                  (while (seq s)
                    (println s n)
                    (if-let [d (read-string (str (last s)))]
                      ((println "d" d)
                       (println s n)
                       (+
                        (if-not (= d 0)
                          (reduce * (repeat n 2))
                          0)
                        (if (> (count s) 1)
                          (bin (butlast s) (inc n))
                          (bin () (inc n))))))))]
          (bin ss 1))) "11"))
(repeat 0 2)
(butlast '(1))
(seq ())
(read-string (str (last "1")))
(take 5 (map #(reduce * (repeat % 2)) (range)))
;=>(1 2 4 8 16)
(map * '(1 2 4) '(1 2 3))
(+ 7 9)
;works!
(fn [ss] (let [s (map #(read-string (str %)) ss)
               t (lazy-seq (map #(reduce * (repeat % 2)) (range)))]
           (reduce + (map * (reverse s) t))))


;--126.
;Through the Looking Class
;; Enter a value which satisfies the following:
;;     (let [x __] (and (= (class x) x) x))
(let [x java.lang.Class]
  (and (= (class x) x) x))
(class identity)
;=>clojure.core$identity
(class clojure.core$identity)
;=>java.lang.Class
(class java.lang.Class)
;=>java.lang.Class
(class Class)
;=>java.lang.Class

;--128.
A standard American deck of playing cards has four suits - spades, hearts, diamonds, 
and clubs - and thirteen cards in each suit. Two is the lowest rank, followed by other integers up to ten; 
then the jack, queen, king, and ace. It's convenient for humans to represent these cards as suit/rank pairs, 
such as H5 or DQ: the heart five and diamond queen respectively. But these forms are not convenient for programmers, 
so to write a card game you need some way to parse an input string into meaningful components. 
For purposes of determining rank, we will define the cards to be valued from 0 (the two) to 12 (the ace) 
Write a function which converts (for example) the string "SJ" into a map of {:suit :spade,:rank 9}. 
A ten will always be represented with the single character "T", rather than the two characters "10".
(= {:suit :diamond :rank 10} (__ "DQ"))
(= {:suit :heart :rank 3} (__ "H5"))
(= {:suit :club :rank 12} (__ "CA"))
(= (range 13) (map (comp :rank __ str)
                   '[S2 S3 S4 S5 S6 S7
                     S8 S9 ST SJ SQ SK SA]))
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
  
 ( in  case [a b]  )
(keep-indexed ranks )
(letfn [(pos [x coll] (keep-indexed (fn [idx val] (when (= val x) idx)) coll))])
(read+string)
(#(case [% %2] 
    9 0 true) 9 0)
;=> true
(type (read-string "H5"))
;; (require '[clojure.spec.alpha :as s])
;; (s/def )


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


(fn f [a b c & s]
  (let [r (b a c)]
    (if (not (empty? s))
      ()
      r)))

(fn f [a & s]
  (if (second s)
    (let [r ((first s) a (second s))]
      (f r s))
    a))

(fn f [a & s]
  (if (second s)
    (let [r
          ((first s)
           a
           (second s))]
      (println "r" r)
      (println "s" s)
      (f r (next (next s))))
    a))

(boolean ())
(next [8])

(not (empty? nil))
(not nil)

((defn f [a [b & s :as g]]
   (println a)
   (println b)
   (println s)
   (println "ss" g)) 1 [2 3 4 5 6 7])

(= 7  ((fn f [a & s]
         (if (second s)
           (let [r
                 ((first s)
                  a
                  (second s))]
             (println "r" r)
             (println "s" s)
             (f r s))
           a)) 2 + 5))
(= 42 ((fn f [a & s]
         (if (second s)
           (let [r
                 ((first s)
                  a
                  (second s))]
             (println "r" r)
             (println "s" s)
             (f r (next (next s))))
           a)) 38 + 48 - 2 / 2))

((fn f [a & s]
   (if (second s)
     (do (println "(second s)" (second s))
         (let [r
               ((first s)
                a
                (second s))]
           (println "r" r)
           (println "s" s)
           (println (next (next s)))
           (f r (next (next s)))))
     a)) 38 + 48 - 2 / 2)

((fn f [a & s]
   (if (second s)
     (do (println "(second s)" (second s))
           ;(println "r" r)
         (println "s" s)
         (println (next (next s)))
         (recur ((first s) a (second s)) (next (next s))))
     a)) 38 + 48 - 2 / 2)

(= 8  ((fn f [a & s]
         (if (second s)
           (do (println "(second s)" (second s))
           ;(println "r" r)
               (println "s" s)
               (println (next (next s)))
               (recur ((first s) a (second s)) (next (next s))))
           a)) 10 / 2 - 1 * 2))
(= 72 ((fn f [a & s]
         (if (second s)
           (do (println "(second s)" (second s))
           ;(println "r" r)
               (println "s" s)
               (println (next (next s)))
               (recur ((first s) a (second s)) (next (next s))))
           a)) 20 / 2 + 2 + 4 + 8 - 6 - 10 * 9))

(second nil)

(fn f [a & s]
  (if (second s)
    (do (println "(second s)" (second s))
           ;(println "r" r)
        (println "s" s)
        (println (next (next s)))
        (recur ((first s) a (second s)) (next (next s))))
    a))
?????
;doesnt work (in cicles, last 3 examples) because of not using recur
(fn f [a & s]
  (if (second s)
    (f ((first s) a (second s)) (next (next s)))
    a))
;works
(fn f [a & s]
  (if (second s)
    (recur ((first s) a (second s)) (next (next s)))
    a))


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
  (letfn
   [(foo [res s]
      (if (associative? s)
        (map (foo (conj res %1) %2) keys vals)))]
    (foo [] ss)))

(associative?)
(= '(1 2) (keys '{1 a
                  2 b}))
;true
(type (keys '{1 a
              2 b}))
;=>clojure.lang.APersistentMap$KeySeq
(type '(1 2))
;=>clojure.lang.PersistentList
(= '(1 2) (seq (keys '{1 a
                       2 a})))


((fn  [ss]
   (letfn
    [(foo [res s]
       (println s)
       (if (and (associative? s) (not (sequential? s)))
         (do (println s)
             (println "res" res)
             (println "fi" (first  (keys s)))
             (map
              #(foo
                (conj res %1) %2)
              (keys s)
              (vals s)))
         (conj res s)))]
     (foo [] ss))) '{a {p 1, q 2}
                     b {m 3, n 4}})
;=>(([a p 1] [a q 2]) ([b m 3] [b n 4]))
;works
((fn  [ss]
   (letfn
    [(foo [res s]
       (println s)
       (if (and (associative? s) (not (sequential? s)))
         (do (println s)
             (println "res" res)
             (println "fi" (first  (keys s)))
             ;(identity ;assoc {} 
             (concat ;=>([a p] 1 [a q] 2 [b m] 3 [b n] 4)
              (mapcat ;=>{[[a p] 1] [[a q] 2], [[b m] 3] [[b n] 4]}
               #(foo
                 (conj res %1) %2)
               (keys s)
               (vals s))))
         ;)         
         ;(merge (apply concat (assoc {} res s)));=>((([a p] 1) ([a q] 2)) (([b m] 3) ([b n] 4)))
         ;(assoc {} res s);=>(({[a p] 1} {[a q] 2}) ({[b m] 3} {[b n] 4}))
         ;(vector res s);=>([a p] 1 [a q] 2 [b m] 3 [b n] 4)
         (vector res s)))]
     (apply assoc {} (foo [] ss)))) '{a {p 1, q 2}
                                      b {m 3, n 4}})
;?????????? how to extract values from collection (list/vector/.. )??????????????
(get)
(doseq [[1 8] [9 5]])
(apply assoc {} '([a p] 1 [a q] 2 [b m] 3 [b n] 4))
(identity [[1 8] [9 5]])
;[[1 8] [9 5]]
(lazy-seq [[1 8] [9 5]])
;=>([1 8] [9 5])
(lazy-cat [[1 8] [9 5]])
;=>([1 8] [9 5])
(reduce concat [[1 8] [9 5]])
;=>(1 8 9 5)
(filter constantly [[1 8] [9 5]])
;=>([1 8] [9 5])
((fn [[a]] ((print a) a)) [[1 8]])
(seq ([a p] 1 [a q] 2 [b m] 3 [b n] 4))
(apply assoc {} '([a p] 1 [a q] 2 [b m] 3 [b n] 4))

(keys '{m {1 [a b c] 3 nil}})
(mapcat vector  '(m) '({7 9}))
(sequential?  '{m {1 [a b c] 3 nil} n {1 [a b c] 3 nil}})

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
(fn a [v]
  (lazy-seq
   (let [t (cons
            (first v)
            (conj
             (map #(apply + %) (partition 2
                                          (mapcat #(repeat 2 %) (rest (butlast v)))))
             (last v)))]
     (vec
      (conj
       t
       (a t))))))
(fn [vv]
  (letfn [(foo [v]
            (lazy-seq
             (let [t (map
                      +
                      (cons
                       (first v)
                       (conj
                        (partition 2 (map #(repeat 2 %) (rest (butlast v))))
                        (last v))))]
               (println "t" t)
               (vec
                (conj
                 t
                 (foo t))))))]
    (foo vv)))
(duplicates [7])
(def v [2 3 2])
(map bigint [2 3 2])
(map
 +
 (cons
  (first v)
  (conj
   (mapcat identity (partition 2 (mapcat #(repeat 2 %) (rest (butlast v)))))
   (last v))))
;+
(concat
 (list (first v))
 (map #(apply + %)
      (partition 2
                 (concat
                  (list (first v))
                  (mapcat #(repeat 2 %) (rest (butlast v)))
                  (list (last v)))))
 (list (last v)))

(fn a [v]
  (lazy-seq
   (let [t (cons
            (first v)
            (conj
             (map #(apply + %) (partition 2
                                          (mapcat #(repeat 2 %) (rest (butlast v)))))
             (last v)))]
     (vec
      (conj
       t
       (a t))))))


(fn [vv]
  (letfn [(foo [v]
            (lazy-seq
             (let [t (map #(apply + %)
                          (partition 2
                                     (concat
                                      (list (first v))
                                      (mapcat #(repeat 2 %) (rest (butlast v)))
                                      (list (last v)))))]
               (println "t" t)
               (vec
                (conj
                 t
                 (foo t))))))]
    (foo vv)))


;?????????how to extract values from coll (list/vector)???????????
(map identity [[7 8 9]])
;=>([7 8 9])
(mapcat identity [[7 8 9]])
;=>(7 8 9)
(def w [7 8 9])
((vec (concat
       (list (first w))
       (map #(apply + %)
            (partition 2
                       (concat
                        (list (first w))
                        (mapcat #(repeat 2 %) (rest (butlast w)))
                        (list (last w)))))
       (list (last w)))) [7 8 9])

(fn [vv]
  (letfn [(expand [w] (vec (concat
                            (list (first w))
                            (map #(apply + %)
                                 (partition 2
                                            (concat
                                             (list (first w))
                                             (mapcat #(repeat 2 %) (rest (butlast w)))
                                             (list (last w)))))
                            (list (last w)))))
          (foo [v]
            (lazy-seq
             (expand v)
             (foo (expand  v))))]
    (lazy-seq (concat
               vv
               (foo vv)))))
;works!
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
;; works!!!!! getting elements (content) from the collections (lists/vectors)
;; ((fn [v] (println v)(let [[a b c] v]  [a b])) [5 6 7])
;=>[5 6]

;--69.
;	Merge with a Function
;; Write a function which takes a function f and a variable number of maps. 
;; Your function should return a map that consists of the rest of the maps conj-ed onto the first. 
;; If a key occurs in more than one map, the mapping(s) from the latter (left-to-right) should be 
;; combined with the mapping in the result by calling (f val-in-result val-in-latter)
;;     (= (__ * {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5})
;;        {:a 4, :b 6, :c 20})
;;     (= (__ - {1 10, 2 20} {1 3, 2 10, 3 15})
;;        {1 7, 2 10, 3 15})
;;     (= (__ concat {:a [3], :b [6]} {:a [4 5], :c [8 9]} {:b [7]})
;;        {:a [3 4 5], :b [6 7], :c [8 9]})
;; Special Restrictions : merge-with
(fn [ff & mm]
  (let [s (list* f mm)
        f (first s)
        m (rest s)]
    (letfn [(foo [v]
              ())])))
(merge)
(contains?)
(boolean (rest []))
(concat {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5})
;=>([:a 2] [:b 3] [:c 4] [:a 2] [:b 2] [:c 5])
(apply concat {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5})
;=>([:a 2] [:b 3] [:c 4] [:a 2] [:b 2] :c 5)
(apply assoc {} (concat {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5}))
;=>{[:a 2] [:b 3], [:c 4] [:a 2], [:b 2] [:c 5]}
(map #(apply assoc {} %) (concat {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5}))
;=>({:a 2} {:b 3} {:c 4} {:a 2} {:b 2} {:c 5})
(map (fn [k v] if (contains? res k)
       (assoc res k (f (k res) v))
       (assoc res k v)))
(fn [f & m]
  (let [ss (map #(apply assoc {} %) (concat m))]
    (letfn
     [(foo [res s]
        (map (fn [k v]
               (if (contains? res k)
                 (assoc res k (f (k res) v))
                 (assoc res k v))) s))]
      (foo '{} ss))))

(= ((fn [f & m]
      (let [ss (map #(apply assoc {} %) (apply concat m))]
        (println ss)
        (letfn
         [(foo [res s]
            (map (fn [[k v]]
                   (println [k v])
                   (if (contains? res k)
                     (assoc res k (f (k res) v))
                     (assoc res k v))) s))]
          (foo {} ss)))) * {:a 2, :b 3, :c 4} {:a 2} {:b 2} {:c 5})
   {:a 4, :b 6, :c 20})
(contains? {} :k)
(map #(println %) '({:a 2} {:b 3} {:c 4} {:a 2} {:b 2} {:c 5}))
(first {:a 2})
(def res {})
(map (fn [[k v]]
       (println [k v])
       (if (contains? res k)
         (assoc res k (+ (k res) v))
         (assoc res k v))) '({:a 2} {:b 3} {:c 4} {:a 2} {:b 2} {:c 5}))
; Error printing return value (UnsupportedOperationException) at clojure.lang.RT/nthFrom (RT.java:992).
; nth not supported on this type: PersistentArrayMap
(map (fn [m]
       (println m)
      ;;  (if (contains? res k)
      ;;    (assoc res k (+ (k res) v))
      ;;    (assoc res k v))
       )'({:a 2} {:b 3} {:c 4} {:a 2} {:b 2} {:c 5}))
(reduce (fn [a b]
          (let [k (apply key b)
                v (k b)]
      ; (println m)
            (if (contains? a k)
              (assoc a k (+ (k a) v))
              (assoc a k v)))) '({:a 2} {:b 3} {:c 4} {:a 2} {:b 2} {:c 5}))
(fn [f & m]
  (let [s (map #(apply assoc {} %) (apply concat m))]
    (reduce (fn [a b]
              (let [k (apply key b)
                    v (k b)]
                (if (contains? a k)
                  (assoc a k (f (k a) v))
                  (assoc a k v))))
            s)))
(first {:a 2});=>[:a 2]
(map key {:a 2});=> (:a)
(apply key {:a 2});=>:a
(:a {:a 2});=>2
(2 {2 20}); java.lang.Long cannot be cast to clojure.lang.IFn
({2 20} 2); =>20
;works
(fn [f & m]
  (let [s (map #(apply assoc {} %) (apply concat m))]
    (println s)
    (reduce (fn [a b]
              (let [k (apply key b)
                    v (b k)]
                (if (contains? a k)
                  (assoc a k (f (a k) v))
                  (assoc a k v))))
            s)))

;--70.
;; Write a function which splits a sentence up into a sorted list of words. 
;; Capitalization should not affect sort order and punctuation should be ignored.
;; (= (__  "Have a nice day.")
;;    ["a" "day" "Have" "nice"])
;; (= (__  "Clojure is a fun language!")
;;    ["a" "Clojure" "fun" "is" "language"])
;; (= (__  "Fools fall for foolish follies.")
;;    ["fall" "follies" "foolish" "Fools" "for"])
("Have a nice day.")
(clojure.string/split "Have a nice day." #"(\b\s)")
;=>["Have" "a" "nice" "day."]
(clojure.string/split "Have a nice day." #"(\W)")
;=>["Have" "a" "nice" "day"]
(str "\u00a0")
(char ".")
(int \.)
(sort (clojure.string/split "Have a nice day." #"(\W)"))
(clojure.string/trim)
(clojure.string/split " something and ACamelName,h ghjf " #"(\s)")
((fn [s]
   (->>
    s
    (#(clojure.string/split % #"(\W)")))) "Have a nice day.")
;=>["Have" "a" "nice" "day"]
((fn [s]
   (->>
    s
    (#(clojure.string/split % #"(\W)"))
    )) "Have a nice day.")
;;=>["Have" "a" "nice" "day"]
((fn [s]
   (->>
    s
    (#(clojure.string/split % #"(\W)"))
    (#(clojure.string/replace % "." ""))
    )) "Have a nice day.")
;=>"[\"Have\" \"a\" \"nice\" \"day\"]"
((fn [s]
   (->>
    s
    (#(clojure.string/replace % "." ""))
    (#(clojure.string/split % #"(\W)"))
    )) "Have a nice day.")
;=>["Have" "a" "nice" "day"]
(sort-by)
(compare "Mae" "a")
;=>-20
(compare "mae" "a")
;=>12
(sort  ["Have" "a" "nice" "day"])
;=>("Have" "a" "day" "nice")
(sort (fn [a b] (neg? (compare a b))) ["Have" "a" "nice" "day"])
;=>("Have" "a" "day" "nice")
(sort (fn [a b] (pos? (compare a b))) ["Have" "a" "nice" "day"])
;=>("nice" "day" "a" "Have")
(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) ["Have" "a" "nice" "day"])
;=>("a" "day" "Have" "nice")
((fn [s]
   (->>
    s
    (#(clojure.string/replace % "." ""))
    (#(clojure.string/split % #"(\W)"))
    (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
    )) "Have a nice day.")
;=>("a" "day" "Have" "nice")
;works!
(fn [s]
  (->>
   s
   (#(clojure.string/replace % "." ""))
   (#(clojure.string/split % #"(\W)"))
   (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
   ))
(= ((fn [s]
      (->>
       s
       (#(clojure.string/replace % "." ""))
       (#(clojure.string/split % #"(\W)"))
       (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
       ))  "Have a nice day.")
   ["a" "day" "Have" "nice"])
(= ((fn [s]
      (->>
       s
       (#(clojure.string/replace % #"\<\(\[\{\\\^\-\=\$\!\|\]\}\)\?\*\+\.\>" ""))
       (#(clojure.string/split % #"(\W)"))
       (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
       ))  "Clojure& is& %a fun$ language!");=>("" "" "" "" "a" "Clojure" "fun" "is" "language")
   ["a" "Clojure" "fun" "is" "language"])
(= (__  "Fools fall for foolish follies.")
   ["fall" "follies" "foolish" "Fools" "for"])
(count "")
(clojure.string/
; "<([{\\^-=$!|]})?*+.>"
; /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]/g 
????wtf  
(= ((fn [s]
      (->>
       s
       (#(clojure.string/replace % #"(^\w\s)" ""))
       (#(clojure.string/split % #"(\W)"))       
       (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
       ))  "Clojure& is& %a fun$ language!");=>("" "" "" "" "a" "Clojure" "fun" "is" "language")
   ["a" "Clojure" "fun" "is" "language"])
 ;works
 (= ((fn [s]
       (->>
        s
        (#(clojure.string/replace % #"(^\w\s)" ""))
        (#(clojure.string/split % #"(\W)"))
        (#(filter (fn [a] (> (count a) 0)) %))
        (#(sort (fn [a b] (neg? (compare (clojure.string/lower-case a) (clojure.string/lower-case b)))) %))
        ))  "Clojure& is& %a fun$ language!");=>("a" "Clojure" "fun" "is" "language")
    ["a" "Clojure" "fun" "is" "language"])
;but error 
;; user=> function(S){for(;;){cf.h(1,arguments.length)||N5(a,A,d,e,Xl(Wo.g(arguments)));var lb=zd(g,Q,S);lb=Y(a,lb,u);if(lb instanceof
;; O5)lb=g4(lb),S=w(lb,0);else return lb}}
;the same:
(fn [s]
  (let [replased (clojure.string/replace s #"(^\w\s)" "")
        splitted (clojure.string/split replased #"(\W)")
        filtered (filter (fn [a] (> (count a) 0)) splitted)
        sorted (sort 
                (fn [a b] 
                  (neg? (compare 
                         (clojure.string/lower-case a) 
                         (clojure.string/lower-case b)))) 
                filtered)]
   sorted))


;--73.
;Analyze a Tic-Tac-Toe Board
;; A tic-tac-toe board is represented by a two dimensional vector. 
;; X is represented by :x, O is represented by :o, and empty is represented by :e. 
;; A player wins by placing three Xs or three Os in a horizontal, vertical, or diagonal row. 
;; Write a function which analyzes a tic-tac-toe board and returns 
;; :x if X has won, :o if O has won, and nil if neither player has won.
;; (= nil (__ [[:e :e :e]
;;             [:e :e :e]
;;             [:e :e :e]]))
;; (= :x (__ [[:x :e :o]
;;            [:x :e :e]
;;            [:x :e :o]]))
;; (= :o (__ [[:e :x :e]
;;            [:o :o :o]
;;            [:x :e :x]]))
;; (= nil (__ [[:x :e :o]
;;             [:x :x :e]
;;             [:o :x :o]]))
;; (= :x (__ [[:x :e :e]
;;            [:o :x :e]
;;            [:o :e :x]]))
;; (= :o (__ [[:x :e :o]
;;            [:x :o :e]
;;            [:o :e :x]]))
;; (= nil (__ [[:x :o :x]
;;             [:x :o :x]
;;             [:o :x :o]]))
(apply map  vector [[:x :o :x]
 [:x :o :x]
 [:o :x :o]])
;;=> ([:x :x :o] 
;;  [:o :o :x] 
;;  [:x :x :o])
(fn [w]
  (let [[a b c] w
        tr (apply map vector w)
        [x y z] tr]
    (letfn
     [(horizontal-win [v]
        (if (some #(every? (fn [q] (= q (first %))) %) w)
          (first (first w))))
      ()])))
(def w [[:x :o :x]
        [:x :x :x]
        [:o :o :o]])
(def result (atom nil))
(if (some #(every? (fn [q] (= q (first %))) %) w)
  (do (swap! conj (first ))
   (first (first w)))
  nil) 
;works
(if (some (fn [p] (every? (fn [q] (= q (first p))) p)) w)  
      (first (first w))
  nil)
;works
(if (some (fn [p] (reset! result (first p))(every? (fn [q] (= q (first p))) p)) w)
  @result
  nil)
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn 
     [(horizontal [v]
        (if (some (fn [p] (reset! result (first p)) (every? (fn [q] (= q (first p))) p)) v)
          @result
          nil)   )
      (diagonal [v]
          (if (=
               (first (first v))
               (second (second v))
               (last (last v)))
            (first (first v))
            nil))]
      (diagonal w)
      (diagonal transponed)
      (horizontal w)
      (horizontal transponed)      
      )))
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (reset! result (first p))
                    (and
                     (not (= (first p) :e))
                     (every? (fn [q] (= q (first p))) p))) v)
          @result
          nil))
      (diagonal [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil))]
      (diagonal w)
      (diagonal transponed)
      (horizontal w)
      (horizontal transponed))))
(for [x [[:x :o :x]
         [:x :x :x]
         [:o :x :o]]
      y x]
  y)
;=>(:x :o :x :x :x :x :o :x :o)
(for [x [[:x :o :x]
         [:x :x :x]
         [:o :x :o]]
      y x
      :let [a (first x)]
      :when (= a y)]
  x)
(type 'a);=>clojure.lang.Symbol
(doto)
(dorun (map #(println "hi" %) ["mum" "dad" "sister"]));=>nil
(map #(println "hi" %) ["mum" "dad" "sister"]);=>(nil nil nil)
(time (dorun (for [x (range 1000) y (range 10000) :when (> x y)] [x y])));=>"Elapsed time: 399.146893 msecs"
(time (dorun (for [x (range 1000) y (range 10000) :while (> x y)] [x y])));=>"Elapsed time: 53.987378 msecs"

(= nil ((fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn 
     [(horizontal [v]
        (if (some (fn [p]
                    (reset! result (first p)) 
                    (and 
                     (not (= (first p) :e))
                     (every? (fn [q] (= q (first p))) p))) v)
          @result
          nil)   )
      (diagonal [v]
         (if (and
              (not (= (first (first v)) :e))
              (=               
               (first (first v))
               (second (second v))
               (last (last v))))
            (first (first v))
            nil))]
      (diagonal w)
      (diagonal transponed)
      (horizontal w)
      (horizontal transponed)      
      ))) [[:e :e :e]
            [:e :e :e]
            [:e :e :e]]))
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (if (not (= (first p) :e))
                      (do
                        (reset! result (first p))
                        (every? #(= % (first p)) p)))) v)
          @result
          nil))
      (diagonal [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil))]
      (diagonal w)
      (diagonal transponed)
      (horizontal w)
      (horizontal transponed))))
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (if (not (= (first p) :e))
                      (do
                        (reset! result (first p))
                        (every? #(= % (first p)) p)))) v)
          @result
          nil))
      (diagonal [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil))]
      (diagonal w)
      (diagonal transponed)
      (horizontal w)
      (horizontal transponed))))

(def result (atom nil))
(def w [[:x :o :x]
        [:x :o :x]
        [:o :x :o]])
(def w [[:e :x :e]
        [:o :o :o]
        [:x :e :x]])
(def w [[:x :e :o]
        [:x :o :e]
        [:o :e :x]])
(vec ( map #(->> %
            reverse
            vec) w))

;works horizontal
(defn horizontal [v]
            (if (some (fn [p]
                        (if (not (= (first p) :e))
                          (do
                            (reset! result (first p))
                            (every? #(= % (first p)) p)))) v)
              @result
              nil))
(horizontal w)

;works
(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (let [fp  (first p)]
                      (when-not (= fp :e)
                        (reset! result fp)
                        (every? #(= % fp) p))))
                  v)
          @result
          nil))
      (diagonal-left [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil))
      (diagonal-right [v]
        (if (and
             (not (= (last (first v)) :e))
             (=
              (last (first v))
              (second (second v))
              (first (last v))))
          (last (first v))
          nil))]      
      (or
       (diagonal-left w)
       (diagonal-right w)
       (horizontal w)
       (horizontal transponed)))))

(fn [w]
  (let [result (atom nil)
        transponed (apply map vector w)
        reversed (vec (map #(->> %
                                 reverse
                                 vec) w))]
    (letfn
     [(horizontal [v]
        (if (some (fn [p]
                    (let [fp  (first p)]
                      (when-not (= fp :e)
                        (reset! result fp)
                        (every? #(= % fp) p))))
                  v)
          @result
          nil))
      (diagonal [v]
        (if (and
             (not (= (first (first v)) :e))
             (=
              (first (first v))
              (second (second v))
              (last (last v))))
          (first (first v))
          nil)) ]
      (or
       (diagonal w)
       (diagonal reversed)
       (horizontal w)
       (horizontal transponed)))))

(= nil ((fn [w]
          (let [result (atom nil)
                transponed (apply map vector w)
                reversed (vec (map #(->> %
                                         reverse
                                         vec) w))]
            (letfn
             [(horizontal [v]
                (if (some (fn [p]
                            (let [fp  (first p)]
                              (when-not (= fp :e)
                                (reset! result fp)
                                (every? #(= % fp) p))))
                          v)
                  @result
                  nil))
              (diagonal [v]
                (if (and
                     (not (= (first (first v)) :e))
                     (=
                      (first (first v))
                      (second (second v))
                      (last (last v))))
                  (first (first v))
                  nil))]
              (or
               (diagonal w)
               (diagonal reversed)
               (horizontal w)
               (horizontal transponed))))) [[:e :e :e]
            [:e :e :e]
            [:e :e :e]]))
(= :x ((fn [w]
         (let [result (atom nil)
               transponed (apply map vector w)
               reversed (vec (map #(->> %
                                        reverse
                                        vec) w))]
           (letfn
            [(horizontal [v]
               (if (some (fn [p]
                           (let [fp  (first p)]
                             (when-not (= fp :e)
                               (reset! result fp)
                               (every? #(= % fp) p))))
                         v)
                 @result
                 nil))
             (diagonal [v]
               (if (and
                    (not (= (first (first v)) :e))
                    (=
                     (first (first v))
                     (second (second v))
                     (last (last v))))
                 (first (first v))
                 nil))]
             (or
              (diagonal w)
              (diagonal reversed)
              (horizontal w)
              (horizontal transponed))))) [[:x :e :o]
           [:x :e :e]
           [:x :e :o]]))
(= :o ((fn [w]
         (let [result (atom nil)
               transponed (apply map vector w)
               reversed (vec (map #(->> %
                                        reverse
                                        vec) w))]
           (letfn
            [(horizontal [v]
               (if (some (fn [p]
                           (let [fp  (first p)]
                             (when-not (= fp :e)
                               (reset! result fp)
                               (every? #(= % fp) p))))
                         v)
                 @result
                 nil))
             (diagonal [v]
               (if (and
                    (not (= (first (first v)) :e))
                    (=
                     (first (first v))
                     (second (second v))
                     (last (last v))))
                 (first (first v))
                 nil))]
             (or
              (diagonal w)
              (diagonal reversed)
              (horizontal w)
              (horizontal transponed))))) [[:e :x :e]
           [:o :o :o]
           [:x :e :x]]))
(= nil ((fn [w]
          (let [result (atom nil)
                transponed (apply map vector w)
                reversed (vec (map #(->> %
                                         reverse
                                         vec) w))]
            (letfn
             [(horizontal [v]
                (if (some (fn [p]
                            (let [fp  (first p)]
                              (when-not (= fp :e)
                                (reset! result fp)
                                (every? #(= % fp) p))))
                          v)
                  @result
                  nil))
              (diagonal [v]
                (if (and
                     (not (= (first (first v)) :e))
                     (=
                      (first (first v))
                      (second (second v))
                      (last (last v))))
                  (first (first v))
                  nil))]
              (or
               (diagonal w)
               (diagonal reversed)
               (horizontal w)
               (horizontal transponed))))) [[:x :e :o]
            [:x :x :e]
            [:o :x :o]]))
(= :x ((fn [w]
         (let [result (atom nil)
               transponed (apply map vector w)
               reversed (vec (map #(->> %
                                        reverse
                                        vec) w))]
           (letfn
            [(horizontal [v]
               (if (some (fn [p]
                           (let [fp  (first p)]
                             (when-not (= fp :e)
                               (reset! result fp)
                               (every? #(= % fp) p))))
                         v)
                 @result
                 nil))
             (diagonal [v]
               (if (and
                    (not (= (first (first v)) :e))
                    (=
                     (first (first v))
                     (second (second v))
                     (last (last v))))
                 (first (first v))
                 nil))]
             (or
              (diagonal w)
              (diagonal reversed)
              (horizontal w)
              (horizontal transponed))))) [[:x :e :e]
           [:o :x :e]
           [:o :e :x]]))
(= :o ((fn [w]
         (let [result (atom nil)
               transponed (apply map vector w)
               reversed (vec (map #(->> %
                                        reverse
                                        vec) w))]
           (letfn
            [(horizontal [v]
               (if (some (fn [p]
                           (let [fp  (first p)]
                             (when-not (= fp :e)
                               (reset! result fp)
                               (every? #(= % fp) p))))
                         v)
                 @result
                 nil))
             (diagonal [v]
               (if (and
                    (not (= (first (first v)) :e))
                    (=
                     (first (first v))
                     (second (second v))
                     (last (last v))))
                 (first (first v))
                 nil))]
             (or
              (diagonal w)
              (diagonal reversed)
              (horizontal w)
              (horizontal transponed))))) [[:x :e :o]
           [:x :o :e]
           [:o :e :x]]))

(= nil ((fn [w]
          (let [result (atom nil)
                transponed (apply map vector w)
                reversed (vec (map #(->> %
                                         reverse
                                         vec) w))]
            (letfn
             [(horizontal [v]
                (if (some (fn [p]
                            (let [fp  (first p)]
                              (when-not (= fp :e)
                                (reset! result fp)
                                (every? #(= % fp) p))))
                          v)
                  @result
                  nil))
              (diagonal [v]
                (if (and
                     (not (= (first (first v)) :e))
                     (=
                      (first (first v))
                      (second (second v))
                      (last (last v))))
                  (first (first v))
                  nil))]
              (or
               (diagonal w)
               (diagonal reversed)
               (horizontal w)
               (horizontal transponed))))) [[:x :o :x]
            [:x :o :x]
            [:o :x :o]]))