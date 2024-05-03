role_reallocation_prompt =  """
Now, we are going to continue the debate. Given the summary of the previous round, shuffle the agent roles
%s
As the mediator of the debate, assign new roles to the other agents. Currently, the roles are: Proposer (P), Opposer (O), Questioner (Q). Assign new roles with the arrow sign. For example, P->O, Q->P, O->Q.
Now, assign new roles:
"""
