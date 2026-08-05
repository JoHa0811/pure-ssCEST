# Reproducible Figures for Pure steady-state CEST

This repository contains simple Python scripts used to recreate selected figures from the publication:

**Pure Steady-State CEST**  
Johannes Hammacher, Christoph Kolbitsch, Patrick Schünke
[DOI](https://pdf.sciencedirectassets.com/271222/1-s2.0-S0730725X25X00084/1-s2.0-S0730725X25001900/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC0aCXVzLWVhc3QtMSJHMEUCIQDxIkrAs3%2Fs4PElwNfkIBMb82uAgBUy%2F4Oey4B5aJy5yQIgQ%2BMR6fS85LkT2YTC72JGYnUSwfmLBeuWVIAqsKDp2UcqvAUI9v%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDAZ9c6WXeLAloQNnhSqQBUmDUHgIK%2FipWkK3zkuUtQTkZug%2B8Bg8%2F1HS11Q3Cd1eTShkmwAkEEh6P%2BcbePF74PovTx51SOCJ%2BNz%2BQc%2FNxM3eXbkpGnfH%2FU5hLH%2F9biFJ85hBX09eplPHQBSWlnyjLmhoICYeacdQC3ETFHme2ufKiKOb3tiBrEIwmuWo6Jdt6xvFaC3J%2FY71%2B%2F%2FnRZjhaNdCsW5ZlBaEgEef6Ks%2FTLRCN%2FlgRZa1ID96w6VoA1Sqm%2FLFu4KJkmElY0atmfB14pNFMEfbJHdCeCDgnEU4lXDRAomy1ruA82ZvEQo183zSIaNKrVAmoTe1ImotgikPVJSSlA7TuWjTIbkpsnsoeLs1WghszQqhp5EFArZ2wQ5X%2FUinZoVWslWQY9WqQUatWlh4ReRpDJaSC9Wm7L53u6aQqO87wywJhFtbJnb%2FPvsu40OxPBUsz6xRfmlYgeIdzksFOvhOGo7dY2ZC%2F7hFfiNHafhJFR8YM3c3QO2zU3XbTZjao3j2hiqXpw2SG8QkK7ZQ4mESe%2BkIMTT3jsVj5XpY1cEw5La%2BtR3lKkTHgNOt5bkVpAg1WW3zB%2BslQj2FLmJekJQ%2BVe2VdF3gp4nKqL1ZIF9cBDkbAm%2FNZPO40vx%2BR0a8wmYNXi5uIBWlk0UWY6PHJEMMmmzXq8zqVhOwR2wA834drNWdSX7FNfPvlDhgX2qrq1dZH0BjnSKrN2V2sMZjhohatFqKm%2FbpAQ3xm9ZvschwaH5QyEwhtN7ZPomVy9dVvsKua18pQP%2BXjH1H7TwdF31Q0BVEe%2Bpc3IWDonLByp9j124yF6smeMQPWvCzz6uJ66USMCUwSF63EEhkkDioWrXS3OoeOLW3DkFISmVYWtdPImts6OT5ODB5Y2tbMPmhwtMGOrEBy6gDwhFRIZkpCNJDNP3G77pZSSAPsAY2rtm%2B1zD%2FGex751vGVLPwpiGmgOBvXFrpdaytJUM0piT1n0nlO%2BSa1HVD7idraFC2HkP7irBvCLUe7e6NflsKEoqEXqKLqNJKRn1KwEK%2FP4HIQZO9tTo1cE7HWhUmI6Jn%2B3h0ENwj3H7%2BhqdGBWl6c%2BevtNMJVAvpfwNfvPrM9iXnsDwX2Arx3xV%2BGY0aTTDU2ri%2FaImxk2sW&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260803T133430Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY76Q2EQ7K%2F20260803%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=3a4cc73b1ddbf430522b41971e2f13763ce6333e1383a47579751e5f02128244&hash=3f231211650708249620b157451bb8180e783adfea5855c184afcee6a6a24fd5&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0730725X25001900&tid=spdf-54869bb8-c457-446c-9847-c8e7bb20eacf&sid=ee60ef8b388677482c792b83285ee64d5858gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=020b05575253075d5200&rr=a255b33edabf717b&cc=de)

## Overview

Each script in this repository is self-contained and can be run independently. When executed, the corresponding figure from the publication will be generated automatically and saved to disk (typically as a `.png` or `.pdf` file).

## Requirements

To install the dependencies, you can run:
```bash
pip install -r requirements.txt
````

## Recreate publication figures
To recreate the respective figure from the publication, simply run the corresponding script.

## Reconstruct raw data
For an example of the reconstruction pipeline, please see reconstrucion_example.py

## Generate Pulseq sequences
To generate example sequences used in the publication, simply run the respective write_CEST* script.
