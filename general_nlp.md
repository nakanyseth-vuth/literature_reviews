# Multilingual Text Processing and Multilingual Lexical Databases

## Multilingual Lexical Resources

## 1. Dictionaries and Other Lexical Data

#### Monolingual:

- **Dictionary**: a collection of words, where each words come with a set of information that describes it, and there's a **implicit method** to access each word.
- **Thesaurus**: a collection of words which are similar, related, or have opposite meanings. Description are not included (this is left to **Dictionary**). It may seen as a network of **Word Senses**

**What is a word?** Word could be very ambigious, for example:

- Composed and Composition: one or two words?
- Stock(v) and Stock(n): one or two entries?

**Some basic vocabulary:**

- **Lemmas**: a set of occurances (e.g. to be ={am, are,...})
- **Lexical unit**: a set of morpho-semantically related lemmas (e.g. {to build, building, builder,....})
- **Vocable**: a set of lexical entries
- **Lexical Entry**: a set of word senses with common etymology(coming from the same origin/lemma) and a unique POS (e.g. {to build(v), building(n), builder(n),...})
- **Word Sense/Acception**: a set of morpho-semantically related lemmas (e.g. {to build, building, builder,...})
- **Polysemy**: a Lexical Entry is polysemic if it has different word senses. e.g. river(n) a. a stream of water; b. flow of liquid; c. (poker) last card in a deal.
- **Homonymy**: Relation between words that are spelled the same but has a different meaning.

## 2 Computational Problems posed by ALL Lexical Data

- **Fun Fact**: the biggest database on the matter is the **'Ethnologe'** database. It currently contains 6912 living languages in the world.
- There are 2 main encodings:

  1. ASCII: defines 127 characters (using 7 bits)
  2. EDCDIC (IBM)

- All the encodings present **a major problem**: they are mutually exclusive one to the other
- Hence, an econding was introduced that is able to represent all currently known script: **UNICODE**

**Some definitions**

- **Character**: abstract entity, considered as atomic, used for writing a language
- **Glyph**: abstract graphical form taken by a characrer in a certain context
- **Font**: Set of concrete graphical forms representation a set of glyphs
- **Code point**: association between a character and an integer (its code), as defined by a norm.
