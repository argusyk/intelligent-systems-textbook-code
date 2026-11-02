class Frame:
    def __init__(self, name, parent=None, **slots):
        self.name, self.parent, self.slots, self.demons = name, parent, dict(slots), {}

    def get(self, slot):
        if slot in self.slots:
            return self.slots[slot]
        if ("if_needed", slot) in self.demons:           # IF-NEEDED demon
            return self.demons[("if_needed", slot)](self)
        return self.parent.get(slot) if self.parent else None   # inheritance

    def set(self, slot, value):
        self.slots[slot] = value
        if ("if_added", slot) in self.demons:            # IF-ADDED demon
            self.demons[("if_added", slot)](self, value)

gte = Frame("GTE", has_combustor=True)
apu = Frame("APU", parent=gte, EGT_max=680, oil_p_min=2.5)
apu.demons[("if_needed", "EGT_margin")] = lambda fr: fr.get("EGT_max") - fr.get("EGT")
unit = Frame("APU-123", parent=apu)
unit.demons = apu.demons                 # the instance receives the prototype's demons
unit.demons[("if_added", "EGT")] = (
    lambda fr, v: print("Warning: overheat!") if v > fr.get("EGT_max") else None)

unit.set("EGT", 705)                     # prints: Warning: overheat!
print(unit.get("has_combustor"), unit.get("EGT_margin"))   # True -25
