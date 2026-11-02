class Frame:
    def __init__(self, name, parent=None, **slots):
        self.name, self.parent, self.slots, self.demons = name, parent, dict(slots), {}

    def get(self, slot):
        if slot in self.slots:
            return self.slots[slot]
        if ("if_needed", slot) in self.demons:           # демон IF-NEEDED
            return self.demons[("if_needed", slot)](self)
        return self.parent.get(slot) if self.parent else None   # успадкування

    def set(self, slot, value):
        self.slots[slot] = value
        if ("if_added", slot) in self.demons:            # демон IF-ADDED
            self.demons[("if_added", slot)](self, value)

gtd = Frame("ГТД", has_combustor=True)
apu = Frame("ДСУ", parent=gtd, EGT_max=680, oil_p_min=2.5)
apu.demons[("if_needed", "EGT_margin")] = lambda fr: fr.get("EGT_max") - fr.get("EGT")
unit = Frame("ДСУ-123", parent=apu)
unit.demons = apu.demons                 # екземпляр отримує демони прототипу
unit.demons[("if_added", "EGT")] = (
    lambda fr, v: print("Увага: перегрів!") if v > fr.get("EGT_max") else None)

unit.set("EGT", 705)                     # друкує: Увага: перегрів!
print(unit.get("has_combustor"), unit.get("EGT_margin"))   # True -25
