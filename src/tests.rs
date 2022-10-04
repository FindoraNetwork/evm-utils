// Copyright 2019-2021 PureStake Inc.
// This file is part of Moonbeam.

// Moonbeam is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Moonbeam is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Moonbeam. If not, see <http://www.gnu.org/licenses/>.

use super::*;
use ethereum_types::{H256, U256};
use crate::data::{IndVerifierKey, Proof1, InputField};

fn u256_repeat_byte(byte: u8) -> U256 {
    let value = H256::repeat_byte(byte);

    U256::from_big_endian(value.as_bytes())
}

#[test]
fn write_bool() {
    let value = true;

    let writer_output = EvmDataWriter::new().write(value).build();

    let mut expected_output = [0u8; 32];
    expected_output[31] = 1;

    assert_eq!(writer_output, expected_output);
}

#[test]
fn read_bool() {
    let value = true;

    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: bool = reader.read().expect("to correctly parse bool");

    assert_eq!(value, parsed);
}

#[test]
fn write_u64() {
    let value = 42u64;

    let writer_output = EvmDataWriter::new().write(value).build();

    let mut expected_output = [0u8; 32];
    expected_output[24..].copy_from_slice(&value.to_be_bytes());

    assert_eq!(writer_output, expected_output);
}

#[test]
fn read_u64() {
    let value = 42u64;
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: u64 = reader.read().expect("to correctly parse u64");

    assert_eq!(value, parsed);
}

#[test]
fn write_u128() {
    let value = 42u128;

    let writer_output = EvmDataWriter::new().write(value).build();

    let mut expected_output = [0u8; 32];
    expected_output[16..].copy_from_slice(&value.to_be_bytes());

    assert_eq!(writer_output, expected_output);
}

#[test]
fn read_u128() {
    let value = 42u128;
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: u128 = reader.read().expect("to correctly parse u128");

    assert_eq!(value, parsed);
}

#[test]
fn write_u256() {
    let value = U256::from(42);

    let writer_output = EvmDataWriter::new().write(value).build();

    let mut expected_output = [0u8; 32];
    value.to_big_endian(&mut expected_output);

    assert_eq!(writer_output, expected_output);
}

#[test]
fn read_u256() {
    let value = U256::from(42);
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: U256 = reader.read().expect("to correctly parse U256");

    assert_eq!(value, parsed);
}

#[test]
fn read_selector() {
    use sha3::{Digest, Keccak256};

    #[precompile_utils_macro::generate_function_selector]
    #[derive(Debug, PartialEq, num_enum::TryFromPrimitive)]
    enum FakeAction {
        Action1 = "action1()",
    }

    let selector = &Keccak256::digest(b"action1()")[0..4];
    let mut reader = EvmDataReader::new(selector);

    assert_eq!(
        reader.read_selector::<FakeAction>().unwrap(),
        FakeAction::Action1
    )
}

#[test]
#[should_panic(expected = "to correctly parse U256")]
fn read_u256_too_short() {
    let value = U256::from(42);
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output[0..31]);
    let _: U256 = reader.read().expect("to correctly parse U256");
}

#[test]
fn write_h256() {
    let mut raw = [0u8; 32];
    raw[0] = 42;
    raw[12] = 43;
    raw[31] = 44;

    let value = H256::from(raw);

    let output = EvmDataWriter::new().write(value).build();

    assert_eq!(&output, &raw);
}

#[test]
fn tmp() {
    let u = U256::from(1_000_000_000);
    println!("U256={:?}", u.0);
}

#[test]
fn read_h256() {
    let mut raw = [0u8; 32];
    raw[0] = 42;
    raw[12] = 43;
    raw[31] = 44;
    let value = H256::from(raw);
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: H256 = reader.read().expect("to correctly parse H256");

    assert_eq!(value, parsed);
}

#[test]
#[should_panic(expected = "to correctly parse H256")]
fn read_h256_too_short() {
    let mut raw = [0u8; 32];
    raw[0] = 42;
    raw[12] = 43;
    raw[31] = 44;
    let value = H256::from(raw);
    let writer_output = EvmDataWriter::new().write(value).build();

    let mut reader = EvmDataReader::new(&writer_output[0..31]);
    let _: H256 = reader.read().expect("to correctly parse H256");
}

#[test]
fn write_address() {
    let value = H160::repeat_byte(0xAA);

    let output = EvmDataWriter::new().write(Address(value)).build();

    assert_eq!(output.len(), 32);
    assert_eq!(&output[12..32], value.as_bytes());
}

#[test]
fn read_address() {
    let value = H160::repeat_byte(0xAA);
    let writer_output = EvmDataWriter::new().write(Address(value)).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Address = reader.read().expect("to correctly parse Address");

    assert_eq!(value, parsed.0);
}

#[test]
fn write_h256_array() {
    let array = vec![
        H256::repeat_byte(0x11),
        H256::repeat_byte(0x22),
        H256::repeat_byte(0x33),
        H256::repeat_byte(0x44),
        H256::repeat_byte(0x55),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();
    assert_eq!(writer_output.len(), 0xE0);

    // We can read this "manualy" using simpler functions since arrays are 32-byte aligned.
    let mut reader = EvmDataReader::new(&writer_output);

    assert_eq!(reader.read::<U256>().expect("read offset"), 32.into());
    assert_eq!(reader.read::<U256>().expect("read size"), 5.into());
    assert_eq!(reader.read::<H256>().expect("read 1st"), array[0]);
    assert_eq!(reader.read::<H256>().expect("read 2nd"), array[1]);
    assert_eq!(reader.read::<H256>().expect("read 3rd"), array[2]);
    assert_eq!(reader.read::<H256>().expect("read 4th"), array[3]);
    assert_eq!(reader.read::<H256>().expect("read 5th"), array[4]);
}

#[test]
fn read_h256_array() {
    let array = vec![
        H256::repeat_byte(0x11),
        H256::repeat_byte(0x22),
        H256::repeat_byte(0x33),
        H256::repeat_byte(0x44),
        H256::repeat_byte(0x55),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Vec<H256> = reader.read().expect("to correctly parse Vec<H256>");

    assert_eq!(array, parsed);
}

#[test]
fn write_u256_array() {
    let array = vec![
        u256_repeat_byte(0x11),
        u256_repeat_byte(0x22),
        u256_repeat_byte(0x33),
        u256_repeat_byte(0x44),
        u256_repeat_byte(0x55),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();
    assert_eq!(writer_output.len(), 0xE0);

    // We can read this "manualy" using simpler functions since arrays are 32-byte aligned.
    let mut reader = EvmDataReader::new(&writer_output);

    assert_eq!(reader.read::<U256>().expect("read offset"), 32.into());
    assert_eq!(reader.read::<U256>().expect("read size"), 5.into());
    assert_eq!(reader.read::<U256>().expect("read 1st"), array[0]);
    assert_eq!(reader.read::<U256>().expect("read 2nd"), array[1]);
    assert_eq!(reader.read::<U256>().expect("read 3rd"), array[2]);
    assert_eq!(reader.read::<U256>().expect("read 4th"), array[3]);
    assert_eq!(reader.read::<U256>().expect("read 5th"), array[4]);
}

#[test]
fn read_u256_array() {
    let array = vec![
        u256_repeat_byte(0x11),
        u256_repeat_byte(0x22),
        u256_repeat_byte(0x33),
        u256_repeat_byte(0x44),
        u256_repeat_byte(0x55),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Vec<U256> = reader.read().expect("to correctly parse Vec<H256>");

    assert_eq!(array, parsed);
}

#[test]
fn write_address_array() {
    let array = vec![
        Address(H160::repeat_byte(0x11)),
        Address(H160::repeat_byte(0x22)),
        Address(H160::repeat_byte(0x33)),
        Address(H160::repeat_byte(0x44)),
        Address(H160::repeat_byte(0x55)),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();

    // We can read this "manualy" using simpler functions since arrays are 32-byte aligned.
    let mut reader = EvmDataReader::new(&writer_output);

    assert_eq!(reader.read::<U256>().expect("read offset"), 32.into());
    assert_eq!(reader.read::<U256>().expect("read size"), 5.into());
    assert_eq!(reader.read::<Address>().expect("read 1st"), array[0]);
    assert_eq!(reader.read::<Address>().expect("read 2nd"), array[1]);
    assert_eq!(reader.read::<Address>().expect("read 3rd"), array[2]);
    assert_eq!(reader.read::<Address>().expect("read 4th"), array[3]);
    assert_eq!(reader.read::<Address>().expect("read 5th"), array[4]);
}

#[test]
fn read_address_array() {
    let array = vec![
        Address(H160::repeat_byte(0x11)),
        Address(H160::repeat_byte(0x22)),
        Address(H160::repeat_byte(0x33)),
        Address(H160::repeat_byte(0x44)),
        Address(H160::repeat_byte(0x55)),
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Vec<Address> = reader.read().expect("to correctly parse Vec<H256>");

    assert_eq!(array, parsed);
}

#[test]
fn read_address_array_size_too_big() {
    let array = vec![
        Address(H160::repeat_byte(0x11)),
        Address(H160::repeat_byte(0x22)),
        Address(H160::repeat_byte(0x33)),
        Address(H160::repeat_byte(0x44)),
        Address(H160::repeat_byte(0x55)),
    ];
    let mut writer_output = EvmDataWriter::new().write(array).build();

    U256::from(6).to_big_endian(&mut writer_output[0x20..0x40]);

    let mut reader = EvmDataReader::new(&writer_output);
    match reader.read::<Vec<Address>>() {
        Ok(_) => panic!("should not parse correctly"),
        Err(ExitError::Other(err)) => {
            assert_eq!(err, "tried to parse H160 out of bounds")
        }
        Err(_) => panic!("unexpected error"),
    }
}

#[test]
fn write_address_nested_array() {
    let array = vec![
        vec![
            Address(H160::repeat_byte(0x11)),
            Address(H160::repeat_byte(0x22)),
            Address(H160::repeat_byte(0x33)),
        ],
        vec![
            Address(H160::repeat_byte(0x44)),
            Address(H160::repeat_byte(0x55)),
        ],
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();
    assert_eq!(writer_output.len(), 0x160);

    writer_output
        .chunks_exact(32)
        .map(H256::from_slice)
        .for_each(|hash| println!("{:?}", hash));

    // We can read this "manualy" using simpler functions since arrays are 32-byte aligned.
    let mut reader = EvmDataReader::new(&writer_output);

    assert_eq!(reader.read::<U256>().expect("read offset"), 0x20.into()); // 0x00
    assert_eq!(reader.read::<U256>().expect("read size"), 2.into()); // 0x20
    assert_eq!(reader.read::<U256>().expect("read 1st offset"), 0x80.into()); // 0x40
    assert_eq!(
        reader.read::<U256>().expect("read 2st offset"),
        0x100.into()
    ); // 0x60
    assert_eq!(reader.read::<U256>().expect("read 1st size"), 3.into()); // 0x80
    assert_eq!(reader.read::<Address>().expect("read 1-1"), array[0][0]); // 0xA0
    assert_eq!(reader.read::<Address>().expect("read 1-2"), array[0][1]); // 0xC0
    assert_eq!(reader.read::<Address>().expect("read 1-3"), array[0][2]); // 0xE0
    assert_eq!(reader.read::<U256>().expect("read 2nd size"), 2.into()); // 0x100
    assert_eq!(reader.read::<Address>().expect("read 2-1"), array[1][0]); // 0x120
    assert_eq!(reader.read::<Address>().expect("read 2-2"), array[1][1]); // 0x140
}

#[test]
fn read_address_nested_array() {
    let array = vec![
        vec![
            Address(H160::repeat_byte(0x11)),
            Address(H160::repeat_byte(0x22)),
            Address(H160::repeat_byte(0x33)),
        ],
        vec![
            Address(H160::repeat_byte(0x44)),
            Address(H160::repeat_byte(0x55)),
        ],
    ];
    let writer_output = EvmDataWriter::new().write(array.clone()).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Vec<Vec<Address>> =
        reader.read().expect("to correctly parse Vec<Vec<Address>>");

    assert_eq!(array, parsed);
}

#[test]

fn write_multiple_arrays() {
    let array1 = vec![
        Address(H160::repeat_byte(0x11)),
        Address(H160::repeat_byte(0x22)),
        Address(H160::repeat_byte(0x33)),
    ];

    let array2 = vec![H256::repeat_byte(0x44), H256::repeat_byte(0x55)];

    let writer_output = EvmDataWriter::new()
        .write(array1.clone())
        .write(array2.clone())
        .build();

    assert_eq!(writer_output.len(), 0x120);

    // We can read this "manualy" using simpler functions since arrays are 32-byte aligned.
    let mut reader = EvmDataReader::new(&writer_output);

    assert_eq!(reader.read::<U256>().expect("read 1st offset"), 0x40.into()); // 0x00
    assert_eq!(reader.read::<U256>().expect("read 2nd offset"), 0xc0.into()); // 0x20
    assert_eq!(reader.read::<U256>().expect("read 1st size"), 3.into()); // 0x40
    assert_eq!(reader.read::<Address>().expect("read 1-1"), array1[0]); // 0x60
    assert_eq!(reader.read::<Address>().expect("read 1-2"), array1[1]); // 0x80
    assert_eq!(reader.read::<Address>().expect("read 1-3"), array1[2]); // 0xA0
    assert_eq!(reader.read::<U256>().expect("read 2nd size"), 2.into()); // 0xC0
    assert_eq!(reader.read::<H256>().expect("read 2-1"), array2[0]); // 0xE0
    assert_eq!(reader.read::<H256>().expect("read 2-2"), array2[1]); // 0x100
}

#[test]
fn read_multiple_arrays() {
    let array1 = vec![
        Address(H160::repeat_byte(0x11)),
        Address(H160::repeat_byte(0x22)),
        Address(H160::repeat_byte(0x33)),
    ];

    let array2 = vec![H256::repeat_byte(0x44), H256::repeat_byte(0x55)];

    let writer_output = EvmDataWriter::new()
        .write(array1.clone())
        .write(array2.clone())
        .build();

    // offset 0x20
    // offset 0x40
    // size 0x60
    // 3 addresses 0xC0
    // size 0xE0
    // 2 H256 0x120
    assert_eq!(writer_output.len(), 0x120);

    let mut reader = EvmDataReader::new(&writer_output);

    let parsed: Vec<Address> = reader.read().expect("to correctly parse Vec<Address>");
    assert_eq!(array1, parsed);

    let parsed: Vec<H256> = reader.read().expect("to correctly parse Vec<H256>");
    assert_eq!(array2, parsed);
}

#[test]
fn test_ind_verifier_key() {
    let (ivk, _, _) = get_proof_data();
    let writer_output = EvmDataWriter::new().write(IndVerifierKey(ivk)).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: IndVerifierKey = reader
        .read()
        .expect("to correctly parse IndexVerifierKey<Fr, MultiPC>");

    let second_output = EvmDataWriter::new().write(parsed).build();

    assert_eq!(writer_output, second_output);
}

#[test]
fn test_proof_serialization() {
    let (_, proof, _) = get_proof_data();
    let writer_output = EvmDataWriter::new().write(Proof1(proof)).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Proof1 = reader
        .read()
        .expect("to correctly parse Proof<Fr, MultiPC>");

    let second_output = EvmDataWriter::new().write(parsed).build();

    assert_eq!(writer_output, second_output);
}

#[test]
fn test_field_serialization() {
    let (_, _, field_arr) = get_proof_data();
    let writer_output = EvmDataWriter::new().write(field_arr.to_vec()).build();

    let mut reader = EvmDataReader::new(&writer_output);
    let parsed: Vec<InputField> = reader
        .read()
        .expect("to correctly parse Proof<Fr, MultiPC>");

    let second_output = EvmDataWriter::new().write(parsed).build();

    assert_eq!(writer_output, second_output);
}

use ark_ff::Field;
use ark_relations::{
    lc,
    r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError},
};
use ark_marlin::{IndexVerifierKey, SimpleHashFiatShamirRng, Marlin, Proof};
use ark_bls12_381::{Fr, Bls12_381};
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::marlin_pc::MarlinKZG10;
use ark_ff::UniformRand;
use ark_std::ops::MulAssign;
use blake2::Blake2s;
use rand_chacha::ChaChaRng;

pub(crate) type MultiPC = MarlinKZG10<Bls12_381, DensePolynomial<Fr>>;
type FS = SimpleHashFiatShamirRng<Blake2s, ChaChaRng>;
type MarlinInst = Marlin<Fr, MultiPC, FS>;

#[derive(Copy, Clone)]
struct Circuit<F: Field> {
    a: Option<F>,
    b: Option<F>,
    num_constraints: usize,
    num_variables: usize,
}

impl<ConstraintF: Field> ConstraintSynthesizer<ConstraintF> for Circuit<ConstraintF> {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<ConstraintF>,
    ) -> Result<(), SynthesisError> {
        let a = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        let b = cs.new_witness_variable(|| self.b.ok_or(SynthesisError::AssignmentMissing))?;
        let c = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a.mul_assign(&b);
            Ok(a)
        })?;
        let d = cs.new_input_variable(|| {
            let mut a = self.a.ok_or(SynthesisError::AssignmentMissing)?;
            let b = self.b.ok_or(SynthesisError::AssignmentMissing)?;

            a.mul_assign(&b);
            a.mul_assign(&b);
            Ok(a)
        })?;

        for _ in 0..(self.num_variables - 3) {
            let _ = cs.new_witness_variable(|| self.a.ok_or(SynthesisError::AssignmentMissing))?;
        }

        for _ in 0..(self.num_constraints - 1) {
            cs.enforce_constraint(lc!() + a, lc!() + b, lc!() + c)?;
        }
        cs.enforce_constraint(lc!() + c, lc!() + b, lc!() + d)?;

        Ok(())
    }
}

pub fn get_proof_data() -> (IndexVerifierKey<Fr, MultiPC>, Proof<Fr, MultiPC>, [InputField; 2]) {
    let rng = &mut ark_std::test_rng();
    let universal_srs = MarlinInst::universal_setup(100, 25, 300, rng).unwrap();

    let a = Fr::rand(rng);
    let b = Fr::rand(rng);
    let mut c = a;
    c.mul_assign(&b);
    let mut d = c;
    d.mul_assign(&b);

    let inputs: [InputField; 2] = [InputField(c), InputField(d)];

    let circ = Circuit {
        a: Some(a),
        b: Some(b),
        num_constraints: 25,
        num_variables: 25,
    };

    let (index_pk, index_vk) = MarlinInst::index(&universal_srs, circ.clone()).unwrap();
    let proof = MarlinInst::prove(&index_pk, circ, rng).unwrap();

    (index_vk, proof, inputs)
}