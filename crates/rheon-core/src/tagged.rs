//! Every float of a value, tagged on its way through a human-readable format.
//!
//! [`crate::humanfloat`] tags one field at a time, which a state cannot rely
//! on: any `f64` a model keeps can be NaN (a reading withheld, a slot not
//! observed, a fit not yet made), and a field nobody annotated reached JSON
//! as `null`, so the bank's export refused the state. [`Ser`] and [`De`]
//! wrap a serializer and a deserializer and tag every `f64` and `f32` that
//! passes through them, wherever it sits, with the spellings
//! [`crate::humanfloat`] writes (`"inf"`, `"-inf"`, `"nan"`, `"-nan"`,
//! `"nan:0x..."`), so a new field or a new model needs nothing.
//!
//! Only the JSON export goes through them: msgpack, the state file, is
//! written and read as before, byte for byte. A field already annotated
//! writes the same text as before, since the annotation tags it first.
//!
//! What the wrapper cannot reach is a float that serde buffers before it
//! knows its type: inside a `#[serde(untagged)]` enum or a
//! `#[serde(flatten)]` field, a tag is read back as a string. Such a field
//! keeps its [`crate::humanfloat`] annotation; one that lacks it is caught,
//! as before, by the export reading itself back.

use serde::de::{self, DeserializeSeed, Deserializer, Visitor};
use serde::ser::{self, Serialize, Serializer};
use std::fmt;

use crate::humanfloat::{tag, untagged};

/// A serializer that writes every non-finite float as its tag.
pub struct Ser<S>(pub S);

/// `value`, serialized through [`Ser`].
struct Tagged<'a, T: ?Sized>(&'a T);

impl<T: ?Sized + Serialize> Serialize for Tagged<'_, T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(Ser(s))
    }
}

/// A compound serializer whose every element goes through [`Ser`].
pub struct Compound<C>(C);

macro_rules! forward_ser {
    ($($m:ident($($a:ident: $t:ty),*);)*) => {
        $(fn $m(self $(, $a: $t)*) -> Result<S::Ok, S::Error> {
            self.0.$m($($a),*)
        })*
    };
}

impl<S: Serializer> Serializer for Ser<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = Compound<S::SerializeSeq>;
    type SerializeTuple = Compound<S::SerializeTuple>;
    type SerializeTupleStruct = Compound<S::SerializeTupleStruct>;
    type SerializeTupleVariant = Compound<S::SerializeTupleVariant>;
    type SerializeMap = Compound<S::SerializeMap>;
    type SerializeStruct = Compound<S::SerializeStruct>;
    type SerializeStructVariant = Compound<S::SerializeStructVariant>;

    forward_ser! {
        serialize_bool(v: bool);
        serialize_i8(v: i8);
        serialize_i16(v: i16);
        serialize_i32(v: i32);
        serialize_i64(v: i64);
        serialize_i128(v: i128);
        serialize_u8(v: u8);
        serialize_u16(v: u16);
        serialize_u32(v: u32);
        serialize_u64(v: u64);
        serialize_u128(v: u128);
        serialize_char(v: char);
        serialize_str(v: &str);
        serialize_bytes(v: &[u8]);
        serialize_none();
        serialize_unit();
        serialize_unit_struct(name: &'static str);
        serialize_unit_variant(name: &'static str, index: u32, variant: &'static str);
    }

    fn serialize_f32(self, v: f32) -> Result<S::Ok, S::Error> {
        if v.is_finite() || !self.0.is_human_readable() {
            self.0.serialize_f32(v)
        } else {
            self.0.serialize_str(&tag(f64::from(v)))
        }
    }

    fn serialize_f64(self, v: f64) -> Result<S::Ok, S::Error> {
        if v.is_finite() || !self.0.is_human_readable() {
            self.0.serialize_f64(v)
        } else {
            self.0.serialize_str(&tag(v))
        }
    }

    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<S::Ok, S::Error> {
        self.0.serialize_some(&Tagged(v))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        v: &T,
    ) -> Result<S::Ok, S::Error> {
        self.0.serialize_newtype_struct(name, &Tagged(v))
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<S::Ok, S::Error> {
        self.0
            .serialize_newtype_variant(name, index, variant, &Tagged(v))
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, S::Error> {
        self.0.serialize_seq(len).map(Compound)
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, S::Error> {
        self.0.serialize_tuple(len).map(Compound)
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, S::Error> {
        self.0.serialize_tuple_struct(name, len).map(Compound)
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, S::Error> {
        self.0
            .serialize_tuple_variant(name, index, variant, len)
            .map(Compound)
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, S::Error> {
        self.0.serialize_map(len).map(Compound)
    }

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, S::Error> {
        self.0.serialize_struct(name, len).map(Compound)
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, S::Error> {
        self.0
            .serialize_struct_variant(name, index, variant, len)
            .map(Compound)
    }

    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

macro_rules! compound {
    ($tr:ident, $($m:ident($($a:ident: $t:ty),*);)*) => {
        impl<C: ser::$tr> ser::$tr for Compound<C> {
            type Ok = C::Ok;
            type Error = C::Error;
            $(fn $m<T: ?Sized + Serialize>(&mut self $(, $a: $t)*, v: &T) -> Result<(), C::Error> {
                self.0.$m($($a,)* &Tagged(v))
            })*
            fn end(self) -> Result<C::Ok, C::Error> {
                self.0.end()
            }
        }
    };
}

compound!(SerializeSeq, serialize_element(););
compound!(SerializeTuple, serialize_element(););
compound!(SerializeTupleStruct, serialize_field(););
compound!(SerializeTupleVariant, serialize_field(););
compound!(SerializeStruct, serialize_field(key: &'static str););
compound!(SerializeStructVariant, serialize_field(key: &'static str););

impl<C: ser::SerializeMap> ser::SerializeMap for Compound<C> {
    type Ok = C::Ok;
    type Error = C::Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, k: &T) -> Result<(), C::Error> {
        self.0.serialize_key(&Tagged(k))
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, v: &T) -> Result<(), C::Error> {
        self.0.serialize_value(&Tagged(v))
    }
    fn end(self) -> Result<C::Ok, C::Error> {
        self.0.end()
    }
}

/// A deserializer that reads a tag back wherever an `f64` or `f32` is
/// asked for.
pub struct De<D>(pub D);

/// A visitor whose every nested deserializer goes through [`De`].
struct Visit<V>(V);

/// A seed whose deserializer goes through [`De`].
struct Seed<T>(T);

/// A sequence, map, enum or variant access whose every element goes
/// through [`De`].
struct Access<A>(A);

/// The visitor of an `f64` or `f32` that is asked for: a number, or a tag.
struct Float<V>(V);

macro_rules! forward_de {
    ($($m:ident($($a:ident: $t:ty),*);)*) => {
        $(fn $m<V: Visitor<'de>>(self $(, $a: $t)*, v: V) -> Result<V::Value, D::Error> {
            self.0.$m($($a,)* Visit(v))
        })*
    };
}

impl<'de, D: Deserializer<'de>> Deserializer<'de> for De<D> {
    type Error = D::Error;

    forward_de! {
        deserialize_any();
        deserialize_bool();
        deserialize_i8();
        deserialize_i16();
        deserialize_i32();
        deserialize_i64();
        deserialize_i128();
        deserialize_u8();
        deserialize_u16();
        deserialize_u32();
        deserialize_u64();
        deserialize_u128();
        deserialize_char();
        deserialize_str();
        deserialize_string();
        deserialize_bytes();
        deserialize_byte_buf();
        deserialize_option();
        deserialize_unit();
        deserialize_unit_struct(name: &'static str);
        deserialize_newtype_struct(name: &'static str);
        deserialize_seq();
        deserialize_tuple(len: usize);
        deserialize_tuple_struct(name: &'static str, len: usize);
        deserialize_map();
        deserialize_struct(name: &'static str, fields: &'static [&'static str]);
        deserialize_enum(name: &'static str, variants: &'static [&'static str]);
        deserialize_identifier();
        deserialize_ignored_any();
    }

    fn deserialize_f32<V: Visitor<'de>>(self, v: V) -> Result<V::Value, D::Error> {
        self.deserialize_f64(v)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, v: V) -> Result<V::Value, D::Error> {
        if self.0.is_human_readable() {
            self.0.deserialize_any(Float(v))
        } else {
            self.0.deserialize_f64(v)
        }
    }

    fn is_human_readable(&self) -> bool {
        self.0.is_human_readable()
    }
}

impl<'de, V: Visitor<'de>> Visitor<'de> for Float<V> {
    type Value = V::Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a number, or \"inf\" / \"-inf\" / \"nan\"")
    }
    fn visit_f64<E: de::Error>(self, x: f64) -> Result<V::Value, E> {
        self.0.visit_f64(x)
    }
    fn visit_f32<E: de::Error>(self, x: f32) -> Result<V::Value, E> {
        self.0.visit_f32(x)
    }
    fn visit_i64<E: de::Error>(self, x: i64) -> Result<V::Value, E> {
        self.0.visit_i64(x)
    }
    fn visit_u64<E: de::Error>(self, x: u64) -> Result<V::Value, E> {
        self.0.visit_u64(x)
    }
    fn visit_i128<E: de::Error>(self, x: i128) -> Result<V::Value, E> {
        self.0.visit_i128(x)
    }
    fn visit_u128<E: de::Error>(self, x: u128) -> Result<V::Value, E> {
        self.0.visit_u128(x)
    }
    fn visit_str<E: de::Error>(self, s: &str) -> Result<V::Value, E> {
        match untagged(s) {
            Some(x) => self.0.visit_f64(x),
            None => Err(E::invalid_value(de::Unexpected::Str(s), &self)),
        }
    }
}

macro_rules! forward_visit {
    ($($m:ident($t:ty);)*) => {
        $(fn $m<E: de::Error>(self, x: $t) -> Result<V::Value, E> {
            self.0.$m(x)
        })*
    };
}

impl<'de, V: Visitor<'de>> Visitor<'de> for Visit<V> {
    type Value = V::Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.expecting(f)
    }

    forward_visit! {
        visit_bool(bool);
        visit_i8(i8);
        visit_i16(i16);
        visit_i32(i32);
        visit_i64(i64);
        visit_i128(i128);
        visit_u8(u8);
        visit_u16(u16);
        visit_u32(u32);
        visit_u64(u64);
        visit_u128(u128);
        visit_f32(f32);
        visit_f64(f64);
        visit_char(char);
        visit_str(&str);
        visit_borrowed_str(&'de str);
        visit_string(String);
        visit_bytes(&[u8]);
        visit_borrowed_bytes(&'de [u8]);
        visit_byte_buf(Vec<u8>);
    }

    fn visit_none<E: de::Error>(self) -> Result<V::Value, E> {
        self.0.visit_none()
    }
    fn visit_unit<E: de::Error>(self) -> Result<V::Value, E> {
        self.0.visit_unit()
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<V::Value, D::Error> {
        self.0.visit_some(De(d))
    }
    fn visit_newtype_struct<D: Deserializer<'de>>(self, d: D) -> Result<V::Value, D::Error> {
        self.0.visit_newtype_struct(De(d))
    }
    fn visit_seq<A: de::SeqAccess<'de>>(self, a: A) -> Result<V::Value, A::Error> {
        self.0.visit_seq(Access(a))
    }
    fn visit_map<A: de::MapAccess<'de>>(self, a: A) -> Result<V::Value, A::Error> {
        self.0.visit_map(Access(a))
    }
    fn visit_enum<A: de::EnumAccess<'de>>(self, a: A) -> Result<V::Value, A::Error> {
        self.0.visit_enum(Access(a))
    }
}

impl<'de, T: DeserializeSeed<'de>> DeserializeSeed<'de> for Seed<T> {
    type Value = T::Value;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<T::Value, D::Error> {
        self.0.deserialize(De(d))
    }
}

impl<'de, A: de::SeqAccess<'de>> de::SeqAccess<'de> for Access<A> {
    type Error = A::Error;
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, A::Error> {
        self.0.next_element_seed(Seed(seed))
    }
    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, A: de::MapAccess<'de>> de::MapAccess<'de> for Access<A> {
    type Error = A::Error;
    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, A::Error> {
        self.0.next_key_seed(Seed(seed))
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, A::Error> {
        self.0.next_value_seed(Seed(seed))
    }
    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}

impl<'de, A: de::EnumAccess<'de>> de::EnumAccess<'de> for Access<A> {
    type Error = A::Error;
    type Variant = Access<A::Variant>;
    fn variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<(T::Value, Self::Variant), A::Error> {
        self.0
            .variant_seed(Seed(seed))
            .map(|(v, rest)| (v, Access(rest)))
    }
}

impl<'de, A: de::VariantAccess<'de>> de::VariantAccess<'de> for Access<A> {
    type Error = A::Error;
    fn unit_variant(self) -> Result<(), A::Error> {
        self.0.unit_variant()
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, A::Error> {
        self.0.newtype_variant_seed(Seed(seed))
    }
    fn tuple_variant<V: Visitor<'de>>(self, len: usize, v: V) -> Result<V::Value, A::Error> {
        self.0.tuple_variant(len, Visit(v))
    }
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value, A::Error> {
        self.0.struct_variant(fields, Visit(v))
    }
}

#[cfg(test)]
#[path = "tagged_tests.rs"]
mod tests;
