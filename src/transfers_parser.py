import re

class Parser:
    
    def parse_market_value(mv):
        mv = mv.replace("Market value at time of transfer","")
        mv = mv.replace("€","").strip()
        if mv.endswith("m"):
            return float(mv[:-1]) * 1_000_000
        if mv.endswith("k"):
            return float(mv[:-1]) * 1_000
        return float(mv) if mv.isdigit() else None

    def parse_age(a):
        a = re.sub(r'^\D+', '', a)
        nums = re.findall(r'\d+', a)
        return f"{nums[0]}y {nums[1]}m {nums[2]}d" if len(nums) == 3 else None

    def parse_contract(s):
        s = re.sub(r'^\D+', '', s)
        return s if s else "0"

    def parse_fee(fee_text):
        if "Loan fee" in fee_text or "loan transfer" in fee_text:
            fee_num = re.findall(r"€([\d\.]+)(m|k)?", fee_text)
            if fee_num:
                val, unit = fee_num[0]
                val = float(val) * (1_000_000 if unit=="m" else 1_000 if unit=="k" else 1)
                return val, 1
        if "End of loan" in fee_text:
            return 'End of loan', 1
        if "free transfer" in fee_text:
            return "Free Transfer", 0
        if "€" in fee_text:
            fee_num = re.findall(r"€([\d\.]+)(m|k)?", fee_text)
            if fee_num:
                val, unit = fee_num[0]
                val = float(val) * (1_000_000 if unit=="m" else 1_000 if unit=="k" else 1)
                return val, 0
        return None, 0

    @classmethod
    def parse_transfer_table(cls, cells_raw):
        data = {
            "season": None, "transfer_date": None,
            "selling_club": None, "buying_club": None,
            "selling_league": None, "buying_league": None,
            "coach_old": None, "coach_new": None,
            "market_value": None, "age": None,
            "contract_left": None, "fee": None, "is_loan": 0
        }
        cells = [c.strip() for c in cells_raw if c.strip()]
        try:
            meta = cells[0]
            data["season"], data["transfer_date"] = meta.replace("Transfer date", "")[6:].split(" -")
            data["selling_club"]  = cells[1]
            data["buying_club"]   = cells[2]
            data["selling_league"] = cells[3]
            data["buying_league"]  = cells[5]
            data['coach_old'] = cells[9]
            data['coach_new'] = cells[11]
        except:
            return None

        for cell in cells:
            if "Market value" in cell:
                data["market_value"] = cls.parse_market_value(cell)
            elif "Age" in cell:
                data["age"] = cls.parse_age(cell)
            elif "Remaining contract" in cell:
                data["contract_left"] = cls.parse_contract(cell)
            elif "Loan fee" in cell or "loan transfer" in cell:
                fee,_ = cls.parse_fee(cell)
                data["fee"] = fee
                data["is_loan"] = 1
            elif cell.startswith("Fee") or "free transfer" in cell:
                fee,_ = cls.parse_fee(cell)
                data["fee"] = fee

        if data["fee"] is None and data["is_loan"] == 0:
            return None

        return data


