(ns www.learndatalogtoday.org)

;--0.1.
;; Find all movies titles in the database.
[:find ?title :where
 [_ :movie/title ?title]]

;--1.0
;; Find the entity ids of movies made in 1987
[:find ?e
 :where
 [?e :movie/year 1987 _]]

;--1.1
;; Find the entity-id and titles of movies in the database
[:find ?e ?title
 :where
 [?e :movie/title ?title]]

;--1.2
;; Find the name of all people in the database
[:find ?person
 :where
 [_ :person/name ?person]]

;--2.0
;; Find movie titles made in 1985
[:find ?title
 :where
 [?e :movie/year 1985 _]
 [?e :movie/title ?title]]

;--2.1
;; What year was "Alien" released?
[:find ?year
 :where
 [?e :movie/title "Alien"]
 [?e :movie/year ?year]]

;--2.2
;; Who directed RoboCop? 
;; You will need to use [<movie-eid> :movie/director <person-eid>] to find the director for a movie.
[:find ?name
 :where
 [?e :movie/title "RoboCop"]
 [?e :movie/director ?p]
 [?p :person/name ?name]]

;--2.3
;; Find directors who have directed Arnold Schwarzenegger in a movie.
[:find ?name
 :where
 [?a :person/name "Arnold Schwarzenegger"]
 [?e :movie/cast ?a]
 [?e :movie/director ?d]
 [?d :person/name ?name]]

;--3.0
;; Find movie title by year
;; Input: 1988
[:find ?title
 :in $ ?year
 :where
 [?e :movie/title ?title]
 [?e :movie/year ?year]]

;--3.1
;; Given a list of movie titles, find the title and the year that movie was released.
;; Input: ["Lethal Weapon" "Lethal Weapon 2" "Lethal Weapon 3"]
[:find ?title ?year
 :in  $ [?title ...]
 :where
 [?e :movie/title ?title]
 [?e :movie/year ?year]]

;--3.2
;; Find all movie ?titles where the ?actor and the ?director has worked together
;; Input1: "Michael Biehn"
;; Input2: "James Cameron"
[:find ?title
 :in $ ?actor ?director
 :where
 [?e :movie/title ?title]
 [?e :movie/cast ?a]
 [?e :movie/director ?d]
 [?d :person/name ?director]
 [?a :person/name ?actor]]

;--3.3
;; Write a query that, given an actor name and a relation with movie-title/rating, 
;; finds the movie titles and corresponding rating for which that actor was a cast member.
;; Input1: "Mel Gibson"
;; Input2: [["Die Hard" 8.3]
;;           ["Alien" 8.5]
;;           ["Lethal Weapon" 7.6]
;;           ["Commando" 6.5]
;;           ["Mad Max Beyond Thunderdome" 6.1]
;;           ["Mad Max 2" 7.6]
;;           ["Rambo: First Blood Part II" 6.2]
;;           ["Braveheart" 8.4]
;;           ["Terminator 2: Judgment Day" 8.6]
;;           ["Predator 2" 6.1]
;;           ["First Blood" 7.6]
;;           ["Aliens" 8.5]
;;           ["Terminator 3: Rise of the Machines" 6.4]
;;           ["Rambo III" 5.4]
;;           ["Mad Max" 7.0]
;;           ["The Terminator" 8.1]
;;           ["Lethal Weapon 2" 7.1]
;;           ["Predator" 7.8]
;;           ["Lethal Weapon 3" 6.6]
;;           ["RoboCop" 7.5]]
[:find ?title ?rating
 :in $ ?actor [[?title ?rating]]
 :where
 [?a :person/name ?actor]
 [?e :movie/title ?title]
 [?e :movie/cast ?a]]

;--4.0
;; What attributes are associated with a given movie.
;; Input: "Commando"
[:find ?attr
 :in $ ?title
 :where
 [?e :movie/title ?title]
 [?e ?a]
 [?a :db/ident ?attr]]

;--4.1
;; Find the names of all people associated with a particular movie (i.e. both the actors and the directors)
;; Input1: "Die Hard"
;; Input2: [:movie/cast :movie/director]
;wrong:
[:find ?name
 :in $ ?title [?attr ...]
 :where
 [?e :movie/title ?title]
 [?e ?attr ?p]
 [?p ?k]
 [?k :db/ident ?name]]
;; =>
;; :person/death
;; :person/name
;; :person/born
;works:
[:find ?name
 :in $ ?title [?attr ...]
 :where
 [?e :movie/title ?title]
 [?e ?attr ?p]
 [?p :person/name ?name]]

;--4.2
;; Find all available attributes, their type and their cardinality. 
;; This is essentially a query to find the schema of the database. 
;; To find all installed attributes you must use the :db.install/attribute attribute. 
;; You will also need to use the :db/valueType and :db/cardinality attributes as well as :db/ident.
[:find ?attr ?type ?card
 :where
 [?e :db.install/attribute ?a]
 [?a :db/ident ?attr]
 [?a :db/valueType ?t]
 [?t :db/ident ?type]
 [?a :db/cardinality ?c]
 [?c :db/ident ?card]]
;; ?attr	?type	?card
;; :db/fn	:db.type/fn	:db.cardinality/one
;; :movie/year	:db.type/long	:db.cardinality/one
;; :db/lang	:db.type/ref	:db.cardinality/one
;; :db/unique	:db.type/ref	:db.cardinality/one
;; :db.sys/reId	:db.type/ref	:db.cardinality/one
;; :fressian/tag	:db.type/keyword	:db.cardinality/one
;; :db.install/partition	:db.type/ref	:db.cardinality/many
;; :movie/sequel	:db.type/ref	:db.cardinality/one
;; :trivia	:db.type/string	:db.cardinality/many
;; :db/cardinality	:db.type/ref	:db.cardinality/one
;; :db/code	:db.type/string	:db.cardinality/one
;; :db/fulltext	:db.type/boolean	:db.cardinality/one
;; :movie/title	:db.type/string	:db.cardinality/one
;; :person/death	:db.type/instant	:db.cardinality/one
;; :db/isComponent	:db.type/boolean	:db.cardinality/one
;; :db.excise/attrs	:db.type/ref	:db.cardinality/many
;; :db/valueType	:db.type/ref	:db.cardinality/one
;; :movie/director	:db.type/ref	:db.cardinality/many
;; :db.install/function	:db.type/ref	:db.cardinality/many
;; :person/name	:db.type/string	:db.cardinality/one
;; :db/doc	:db.type/string	:db.cardinality/one
;; :db/txInstant	:db.type/instant	:db.cardinality/one
;; :db/excise	:db.type/ref	:db.cardinality/one
;; :db.alter/attribute	:db.type/ref	:db.cardinality/many
;; :db.install/attribute	:db.type/ref	:db.cardinality/many
;; :db.sys/partiallyIndexed	:db.type/boolean	:db.cardinality/one
;; :db/ident	:db.type/keyword	:db.cardinality/one
;; :movie/cast	:db.type/ref	:db.cardinality/many
;; :db.excise/beforeT	:db.type/long	:db.cardinality/one
;; :db/index	:db.type/boolean	:db.cardinality/one
;; :person/born	:db.type/instant	:db.cardinality/one
;; :db/noHistory	:db.type/boolean	:db.cardinality/one
;; :db.install/valueType	:db.type/ref	:db.cardinality/many
;; :db.excise/before	:db.type/instant	:db.cardinality/one

;--4.3
;; When was the seed data imported into the database? 
;; Grab the transaction of any datom in the database, e.g., [_ :movie/title _ ?tx] and work from there.
;wrong:
[:find ?timestamp
 :where
 [?e :db.install/attribute ?a]
 [?a :db/ident ?attr]
 [?i ?attr ?v]
 [?v :db/ident _ ?tx]
 [?tx :db/txInstant ?timestamp]]
;; => processing rule: (q__166 ?timestamp), message: processing clause: [?v :db/ident _ ?tx], message: Cannot resolve key: Joanne Samuel
;wrong:
[:find ?timestamp
 :where
 [?e :db.install/attribute ?a]
 [?a :db/ident ?attr]
 [_ ?attr _ ?tx]
 [?tx :db/txInstant ?timestamp]]
;; => all timestamps:
;; #inst "2022-02-14T06:28:28.904-00:00"
;; #inst "1970-01-01T00:00:00.000-00:00"
;; #inst "2022-02-14T06:28:28.925-00:00"
;works:
[:find ?timestamp
 :where
 [_ :movie/title _ ?tx]
 [?tx :db/txInstant ?timestamp]]
;; #inst "2022-02-14T06:28:28.925-00:00"

;--5.0
;; Find movies older than a certain year (inclusive)
;; Input: 1979
[:find ?title
 :in $ ?year
 :where
 [?e :movie/year ?y]
 [?e :movie/title ?title]
 [(<= ?y ?year)]]

;--5.1
;; Find actors older than Danny Glover
[:find ?name
 :where
 [?pdg :person/name "Danny Glover"]
 [?pdg :person/born ?dgb]
 [(< ?y ?dgb)]
 [_ :movie/cast ?a]
 [?a :person/born ?y]
 [?a :person/name ?name]]

;--5.2
;; Find movies newer than ?year (inclusive) and has a ?rating higher than the one supplied
;; Input1: 1990
;; Input2: 8.0
;; Input3: [["Die Hard" 8.3]
;;    ["Alien" 8.5]
;;    ["Lethal Weapon" 7.6]
;;    ["Commando" 6.5]
;;    ["Mad Max Beyond Thunderdome" 6.1]
;;    ["Mad Max 2" 7.6]
;;    ["Rambo: First Blood Part II" 6.2]
;;    ["Braveheart" 8.4]
;;    ["Terminator 2: Judgment Day" 8.6]
;;    ["Predator 2" 6.1]
;;    ["First Blood" 7.6]
;;    ["Aliens" 8.5]
;;    ["Terminator 3: Rise of the Machines" 6.4]
;;    ["Rambo III" 5.4]
;;    ["Mad Max" 7.0]
;;    ["The Terminator" 8.1]
;;    ["Lethal Weapon 2" 7.1]
;;    ["Predator" 7.8]
;;    ["Lethal Weapon 3" 6.6]
;;    ["RoboCop" 7.5]]
[:find ?title
 :in $ ?minyear ?minrating [[?title ?rating]]
 :where
 [(> ?rating ?minrating)]
 [?e :movie/title ?title]
 [?e :movie/year ?y]
 [(>= ?y ?minyear)]]

;--6.0
;; Find people by age. Use the function tutorial.fns/age to find the age given a birthday and a date representing "today" .
;; Input1: 63
;; Input2: #inst "2013-08-02T00:00:00.000-00:00"
;; (defn tutorial.fns/age [birthday today]
;;   (quot (- (.getTime today)
;;            (.getTime birthday))
;;         (* 1000 60 60 24 365)))
[:find ?name
 :in $ ?age ?today
 :where
[?p :person/name ?name]
[?p :person/born ?born]
[(tutorial.fns/age ?born ?today) ?age]]

;--6.1
;; Find people younger than Bruce Willis and their ages.
;; Input1: #inst "2013-08-02T00:00:00.000-00:00"
[:find ?name ?age
 :in $ ?today
 :where
 [?bu :person/name "Bruce Willis"]
 [?bu :person/born ?buborn]
[?p :person/name ?name]
[?p :person/born ?born]
  [(> ?born ?buborn)]
[(tutorial.fns/age ?born ?today) ?age]
 ]

;--6.2
;; The birthday paradox states that in a room of 23 people there is a 50% chance that someone has the same birthday. 
;; Write a query to find who has the same birthday. Use the < predicate on the names to avoid duplicate answers. 
;; You can use (the deprecated) .getDate and .getMonth java Date methods.
[:find ?name-1 ?name-2
 :where
[?p-1 :person/born ?born-1]
[?p-2 :person/born ?born-2]
[?p-1 :person/name ?name-1]
[?p-2 :person/name ?name-2]
 [(not= ?p-1 ?p-2)] 
 [(.getDate ?born-1) ?date-born-1]
 [(.getDate ?born-2) ?date-born-2]
 [(.getMonth ?born-1) ?date-month-1]
 [(.getMonth ?born-2) ?date-month-2]
 [(= ?date-born-1 ?date-born-2)]
 [(= ?date-month-1 ?date-month-2)]
 [(< ?name-1 ?name-2)]]

;--7.0
;; count the number of movies in the database
[:find (count ?e)
 :where
 [?e :movie/title _]]

;--7.1
;; Find the birth date of the oldest person in the database.
[:find (min ?a)
 :where
 [_ :person/born ?a]]

;--7.2
;; Given a collection of actors and (the now familiar) ratings data.
;; Find the average rating for each actor. 
;; The query should return the actor name and the avg rating.
;; Input1: ["Sylvester Stallone" "Arnold Schwarzenegger" "Mel Gibson"]
;; Input2: [["Die Hard" 8.3]
            ;; ["Alien" 8.5]
            ;; ["Lethal Weapon" 7.6]
            ;; ["Commando" 6.5]
            ;; ["Mad Max Beyond Thunderdome" 6.1]
            ;; ["Mad Max 2" 7.6]
            ;; ["Rambo: First Blood Part II" 6.2]
            ;; ["Braveheart" 8.4]
            ;; ["Terminator 2: Judgment Day" 8.6]
            ;; ["Predator 2" 6.1]
            ;; ["First Blood" 7.6]
            ;; ["Aliens" 8.5]
            ;; ["Terminator 3: Rise of the Machines" 6.4]
            ;; ["Rambo III" 5.4]
            ;; ["Mad Max" 7.0]
            ;; ["The Terminator" 8.1]
            ;; ["Lethal Weapon 2" 7.1]
            ;; ["Predator" 7.8]
            ;; ["Lethal Weapon 3" 6.6]
            ;; ["RoboCop" 7.5]]
[:find ?actor (avg ?rating)
 :in $ [?actor ...] [[?title ?rating]]
 :where
 [?e :movie/cast ?p]
 [?p :person/name ?actor]
 [?e :movie/title ?title]]

;--8.0
;; Write a rule [movie-year ?title ?year] where ?title is the title of some movie and ?year is that movies release year.
;; Query
[:find ?title
 :in $ %
 :where
 [movie-year ?title 1991]]
;; Rules:
[[(movie-year ?title ?year) 
   [?e :movie/title ?title]
   [?e :movie/year ?year]]]

;--8.1
;; Two people are friends if they have worked together in a movie. Write a rule [friends ?p1 ?p2] where p1 and p2 are person entities. 
;; Try with a few different ?name inputs to make sure you got it right. There might be some edge cases here.
;; Input1: "Sigourney Weaver"
;; Query
[:find ?friend
 :in $ % ?name
 :where
 [?p1 :person/name ?name]
 (friends ?p1 ?p2)
 [?p2 :person/name ?friend]]
;; Rules:
[[(friends ?p1 ?p2)
  [?e :movie/cast ?p1]
  [?e :movie/cast ?p2]
  [(not= ?p1 ?p2)]]
 [(friends ?p1 ?p2)
  [?e :movie/cast ?p1]
  [?e :movie/director ?p2]
  [(not= ?p1 ?p2)]]]

;--8.2
Write a rule [sequels ?m1 ?m2] where ?m1 and ?m2 are movie entities. You'll need to use the attribute :movie/sequel. 
To implement this rule correctly you can think of the problem like this: A movie ?m2 is a sequel of ?m1 if either
    ?m2 is the "direct" sequel of m1 or
    ?m2 is the sequel of some movie ?m and that movie ?m is the sequel to ?m1.
There are (at least) three different ways to write the above query. Try to find all three solutions.
;; Input1: "Mad Max"
;; Query
[:find ?sequel
 :in $ % ?title
 :where
 [?m :movie/title ?title]
 (sequels ?m ?s)
 [?s :movie/title ?sequel]]
;; Rules:
[[(sequels ?m1 ?m2)
  [?m1 :movie/sequel ?m2]]
 [(sequels ?m1 ?m2)
  [?m1 :movie/sequel ?m]
  [?m :movie/sequel ?m2]]]